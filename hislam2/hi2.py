import os
import torch
import numpy as np
from lietorch import SE3

from modules.droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from track_frontend import TrackFrontend, resolve_scal3r_frontend_prior_policy
from track_backend import TrackBackend
from util.trajectory_filler import PoseTrajectoryFiller
from util.utils import load_config

from collections import OrderedDict
from torch.multiprocessing import Process, Queue
from gs_backend import GSBackEnd
from pgo_buffer import PGOBuffer


_USE_PUBLISHED_FRAME_ID = object()
_ONLINE_FALLBACK = object()


def _run_online_operation(
    manager,
    phase,
    operation,
    *,
    frame_ids=None,
    chunk_ids=None,
):
    try:
        return operation()
    except Exception as exc:
        if getattr(exc, "_ffgs_online_failure_recorded", False):
            raise
        classifier = getattr(manager, "is_recoverable_integration_error", None)
        try:
            recoverable = bool(classifier(exc)) if callable(classifier) else False
        except Exception as classification_exc:
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"online failure classification also failed: {classification_exc}")
            raise exc.with_traceback(exc.__traceback__) from classification_exc
        reason = str(getattr(exc, "reason", "") or exc)
        recorder = getattr(manager, "record_integration_outcome", None)
        if not callable(recorder):
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note("online manager cannot persist integration failure attribution")
            raise
        try:
            recorder(
                phase=phase,
                outcome="fallback" if recoverable else "fatal",
                recoverable=recoverable,
                reason=reason,
                exception_class=type(exc).__name__,
                message=str(exc),
                frame_ids=[] if frame_ids is None else list(frame_ids),
                chunk_ids=[] if chunk_ids is None else list(chunk_ids),
                owner="hi2",
            )
        except Exception as attribution_exc:
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"online failure attribution also failed: {attribution_exc}")
            raise exc.with_traceback(exc.__traceback__) from attribution_exc
        if recoverable:
            return _ONLINE_FALLBACK
        raise


def _record_online_fallback(manager, *, phase, reason, frame_ids=None, chunk_ids=None, message=None):
    recorder = getattr(manager, "record_integration_outcome", None)
    if not callable(recorder):
        raise RuntimeError("online manager cannot persist fallback attribution")
    return recorder(
        phase=phase,
        outcome="fallback",
        recoverable=True,
        reason=str(reason),
        exception_class=None,
        message=message,
        frame_ids=[] if frame_ids is None else list(frame_ids),
        chunk_ids=[] if chunk_ids is None else list(chunk_ids),
        owner="hi2",
    )


class Hi2:
    def __init__(self, args):
        super(Hi2, self).__init__()
        self.load_weights(args.weights)
        config = load_config(args.config)
        _apply_online_scal3r_cli_overrides(config, args)
        self.config = config
        self.args = args
        self.images = {}

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(config, args.image_size, args.buffer)
        frontend_config = config["Tracking"]["frontend"]
        frontend_prior_policy = resolve_scal3r_frontend_prior_policy(frontend_config)
        self.online_scal3r_manager = self._build_online_scal3r_manager(config, args.output)
        self.scal3r_prior_provider = self._build_scal3r_prior_provider(
            frontend_config,
            frontend_prior_policy,
        )
        self.video.use_scal3r_prior = frontend_prior_policy["active"]

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(
            self.net,
            self.video,
            config["Tracking"]["motion_filter"],
            prior_provider=self.scal3r_prior_provider,
        )

        # frontend process
        self.frontend = TrackFrontend(
            self.net,
            self.video,
            frontend_config,
            scal3r_prior_policy=frontend_prior_policy,
        )

        # backend process
        self.backend = TrackBackend(self.net, self.video, config["Tracking"]["backend"])

        # 3dgs
        self.gs = GSBackEnd(
            config,
            self.args.output,
            args.gsvis,
            online_scal3r_manager=self.online_scal3r_manager,
        )

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        # visualizer
        if args.droidvis:
            from util.droid_visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

        # global PGBA backend
        self.pgba = config["Tracking"]["pgba"]["active"]
        if self.pgba:
            self.video.pgobuf = PGOBuffer(self.net, self.video, self.frontend, config["Tracking"]["pgba"])
            self.LC_data_queue = Queue()
            self.video.pgobuf.set_LC_data_queue(self.LC_data_queue)
            self.mp_backend = Process(target=self.video.pgobuf.spin)
            self.mp_backend.start()

    def load_weights(self, weights):
        """ load trained model weights """
        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()
        ])
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()

    def _build_scal3r_prior_provider(self, frontend_config, frontend_prior_policy):
        if frontend_prior_policy["online"]:
            print(
                "[Online Scal3R] frontend depth-prior consumer enabled "
                f"(effective gate, min confidence {frontend_prior_policy['min_confidence']})"
            )
            return self.online_scal3r_manager.depth_prior_provider(consumer="frontend")

        if not frontend_prior_policy["active"]:
            return None
        scal3r_config = frontend_config.get("scal3r_prior", {})
        from ffgs_slam.prior.hislam2_provider import Scal3RDepthPriorProvider

        provider = Scal3RDepthPriorProvider.from_config(scal3r_config)
        print("[Scal3R prior] enabled")
        return provider

    def _build_online_scal3r_manager(self, config, output_path):
        from ffgs_slam.online import OnlineScal3RManager

        return OnlineScal3RManager.from_config(config, output_path=output_path)

    def call_gs(self, viz_idx, dposes=None, dscale=None, consumer_phase="live"):
        data = {
            'viz_idx':  viz_idx.to(device='cpu'),
            'tstamp':   self.video.tstamp[viz_idx].to(device='cpu'),
            'poses':    self.video.poses[viz_idx].to(device='cpu'),
            'images':   self.video.images[viz_idx.cpu()],
            'normals':  self.video.normals[viz_idx.cpu()],
            'depths':   1./self.video.disps_up[viz_idx.cpu()],
            'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu') * 8,
            'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
            'scale_updates': dscale.to(device='cpu') if dscale is not None else None,
            'consumer_phase': str(consumer_phase),
        }
        self.gs.process_track_data(data)

    def track(self, tstamp, image, intrinsics=None, is_last=False):
        """ main thread - update map """

        with torch.no_grad():
            self.images[tstamp] = image

            # check there is enough motion
            # Motion filtering and feature extraction
            # Keyframe decision
            self.filterx.track(tstamp, image, intrinsics, is_last)

            # local bundle adjustment
            # Frontend local BA
            viz_idx = self.frontend(is_last=is_last)

            self._publish_online_scal3r_keyframes(current_frame_id=tstamp)

        if len(viz_idx):
            self._block_until_online_prior_ready(viz_idx, current_frame_id=tstamp)

        # Optional PGBA
        if len(viz_idx) and self.pgba:
            dposes, dscale = self.video.pgobuf.run_pgba(self.LC_data_queue)
            if dposes is not None:
                self.call_gs(
                    torch.arange(0, self.video.counter.value-1, device='cuda'), 
                    dposes[:-1], 
                    dscale[:-1]
                )

        # If frontend returns updated keyframe indices
        if len(viz_idx):
            self.call_gs(viz_idx)

    def _publish_online_scal3r_keyframes(self, current_frame_id=None, policy_current_frame_id=_USE_PUBLISHED_FRAME_ID):
        manager = self.online_scal3r_manager
        if manager is None:
            return

        context_frame_ids = _run_online_operation(
            manager,
            "keyframe_publication",
            lambda: _online_context_frame_ids(current_frame_id),
        )
        if context_frame_ids is _ONLINE_FALLBACK:
            return None
        control = _run_online_operation(
            manager,
            "keyframe_publication",
            lambda: _online_publication_control(
                self.video,
                current_frame_id,
                policy_current_frame_id,
            ),
            frame_ids=context_frame_ids,
        )
        if control is _ONLINE_FALLBACK:
            return None
        counter, poll_frame_id = control
        if counter <= 0:
            sync_live_keyframes = getattr(manager, "sync_live_keyframes", None)
            if callable(sync_live_keyframes):
                sync_result = _run_online_operation(
                    manager,
                    "keyframe_publication",
                    lambda: sync_live_keyframes([]),
                    frame_ids=context_frame_ids,
                )
                if sync_result is _ONLINE_FALLBACK:
                    return None
            _run_online_operation(
                manager,
                "keyframe_publication",
                lambda: manager.poll(current_frame_id=poll_frame_id),
                frame_ids=context_frame_ids,
            )
            return None

        snapshot = _run_online_operation(
            manager,
            "dba_snapshot",
            lambda: _online_keyframe_snapshot(self.video, counter),
            frame_ids=context_frame_ids,
        )
        if snapshot is _ONLINE_FALLBACK:
            return None
        poses_w2c, tstamps, intrinsics, images, dba_depths, live_frame_ids = snapshot
        sync_live_keyframes = getattr(manager, "sync_live_keyframes", None)
        if callable(sync_live_keyframes):
            sync_result = _run_online_operation(
                manager,
                "keyframe_publication",
                lambda: sync_live_keyframes(live_frame_ids),
                frame_ids=live_frame_ids,
            )
            if sync_result is _ONLINE_FALLBACK:
                return None

        for local_index in range(counter):
            frame_id = live_frame_ids[local_index]
            publish_policy_frame_id = (
                frame_id
                if policy_current_frame_id is _USE_PUBLISHED_FRAME_ID
                else poll_frame_id
            )
            report = _run_online_operation(
                manager,
                "keyframe_publication",
                lambda: manager.publish_keyframe(
                    frame_id=frame_id,
                    keyframe_index=local_index,
                    image=images[local_index],
                    intrinsics=intrinsics[local_index],
                    pose_w2c=poses_w2c[local_index],
                    dba_depth=dba_depths[local_index],
                    metadata={"source": "hislam2_depth_video"},
                    policy_current_frame_id=publish_policy_frame_id,
                ),
                frame_ids=[frame_id],
            )
            if report is _ONLINE_FALLBACK:
                return None
            scheduled_chunk_ids = report.get("scheduled_chunks", [])
            if scheduled_chunk_ids and getattr(manager.config, "wait_for_keyframe_inference", False):
                wait_report = _run_online_operation(
                    manager,
                    "keyframe_wait",
                    lambda: manager.wait_for_scheduled_chunks(
                        scheduled_chunk_ids,
                        current_frame_id=publish_policy_frame_id,
                    ),
                    frame_ids=[frame_id],
                    chunk_ids=scheduled_chunk_ids,
                )
                if wait_report is _ONLINE_FALLBACK:
                    return None
                wait_summary = _online_scal3r_wait_summary(wait_report)
                print(
                    "[Online Scal3R] live keyframe inference "
                    f"chunks={scheduled_chunk_ids} "
                    f"settled={wait_report['settled_chunk_ids']} "
                    f"{wait_summary}"
                )
        _run_online_operation(
            manager,
            "keyframe_publication",
            lambda: manager.poll(current_frame_id=poll_frame_id),
            frame_ids=live_frame_ids,
        )
        return None

    def _block_until_online_prior_ready(self, viz_idx, current_frame_id=None):
        manager = self.online_scal3r_manager
        if manager is None or not manager.config.block_until_prior_ready:
            return None

        context_frame_ids = _run_online_operation(
            manager,
            "blocking_wait",
            lambda: _online_context_frame_ids(current_frame_id),
        )
        if context_frame_ids is _ONLINE_FALLBACK:
            return None
        frame_ids = _run_online_operation(
            manager,
            "blocking_wait",
            lambda: _frame_ids_for_viz_idx(self.video, viz_idx),
            frame_ids=context_frame_ids,
        )
        if frame_ids is _ONLINE_FALLBACK:
            return None
        if not frame_ids:
            return None

        report = _run_online_operation(
            manager,
            "blocking_wait",
            lambda: manager.block_until_prior_ready(
                frame_ids=frame_ids,
                current_frame_id=_to_int_or_none(current_frame_id),
            ),
            frame_ids=frame_ids,
        )
        if report is _ONLINE_FALLBACK:
            return None
        print(
            "[Online Scal3R] blocking wait "
            f"targets={frame_ids} ready={report['ready_frame_ids']} "
            f"timed_out={report['timed_out']} "
            f"{_online_scal3r_wait_summary(report)}"
        )
        if not report["ready_frame_ids"]:
            reason = "blocking_wait_timeout" if report["timed_out"] else "prior_not_ready"
            _record_online_fallback(
                manager,
                phase="blocking_wait",
                reason=reason,
                frame_ids=frame_ids,
                message=(
                    f"No ready online prior for frames {frame_ids}; "
                    f"timed_out={report['timed_out']}"
                ),
            )
        return report

    def terminate(self):
        """ terminate the visualization process, return poses [t, q] """
        self.video.ready.value = 1
        if self.pgba:
            dposes, dscale = self.video.pgobuf.run_pgba(self.LC_data_queue)
            if dposes is not None:
                self.call_gs(
                    torch.arange(0, self.video.counter.value, device='cuda'),
                    dposes, 
                    dscale,
                    consumer_phase="termination",
                )
            self.mp_backend.terminate()
        del self.frontend

        # check if new keyframes need to be added
        deltas = np.add.accumulate(self.filterx.deltas)
        d_covis = self.video.distance_covis(torch.arange(1, self.video.counter.value-1, device='cuda'))
        new_kfs = []
        for i in torch.arange(1, self.video.counter.value-1, device='cuda'):
            if d_covis[i-1] > self.config['Tracking']['backend']['covis_thresh']:
                delta = deltas[int(self.video.tstamp[i-1])] + (deltas[int(self.video.tstamp[i])] - deltas[int(self.video.tstamp[i-1])]) / 2
                ind1 = np.where(deltas > delta)[0][0]
                if ind1 not in self.video.tstamp:
                    new_kfs.append(ind1)
                delta = deltas[int(self.video.tstamp[i])] + (deltas[int(self.video.tstamp[i+1])] - deltas[int(self.video.tstamp[i])]) / 2
                ind2 = np.where(deltas > delta)[0][0]
                if ind2 not in self.video.tstamp:
                    new_kfs.append(ind2)
                print(f' - add new keyframe {ind1} and {ind2} for {self.video.tstamp[i].item()}')
        new_kfs = sorted(list(set(new_kfs)))

        # fill in poses for new keyframes
        for i in range(0, len(new_kfs), 10):
            new_kf = new_kfs[i:i+10]
            images = [self.images[i] for i in new_kf]
            Gs, gmap = self.traj_filler.fill(new_kf, images, return_fmap=True)
            inputs = torch.stack(images).cuda() / 255.0
            inputs = inputs.sub_(self.filterx.MEAN).div_(self.filterx.STDV)
            net, inp = self.filterx.context_encoder(inputs[:,[0]])
            for i, ind in enumerate(new_kf):
                place = (self.video.tstamp > ind).nonzero()[0].item()
                self.video.shift(place)
                depth, normal = self.filterx.prior_extractor(inputs[i])
                depth, prior_conf = self.filterx.apply_external_prior(
                    ind,
                    depth,
                    consumer_phase="termination",
                )
                self.video[place] = (
                    ind, 
                    images[i], 
                    Gs.data[i], 
                    self.video.disps[place].mean(), 
                    depth.cpu(), 
                    normal.cpu(), 
                    None, 
                    gmap[i], 
                    net[i,0], 
                    inp[i,0],
                    prior_conf.cpu() if prior_conf is not None else None
                )
        del self.filterx

        # global bundle adjustment
        poses_pre = self.video.poses[:self.video.counter.value].clone()
        self.backend(4)
        self.backend(8)
        del self.backend
        poses_pos = self.video.poses[:self.video.counter.value].clone()
        dposes = SE3(poses_pos).inv() * SE3(poses_pre)
        dscale = torch.ones(self.video.counter.value, 1)
        torch.cuda.empty_cache()

        # final refinement
        self.call_gs(
            torch.arange(0, self.video.counter.value, device='cuda'), 
            dposes, 
            dscale,
            consumer_phase="termination",
        )
        updated_poses = self.gs.finalize()
        self.video.poses[:self.video.counter.value] = torch.tensor(updated_poses[:,1:])

        traj_full = self.traj_filler(self.images)
        self.gs.eval_rendering(
            self.images, 
            self.args.gtdepthdir, 
            traj_full.matrix().data, 
            self.video.tstamp[:self.video.counter.value].to(device='cpu'),
            save_render_depth=self.args.save_render_depth
        )
        if self.online_scal3r_manager is not None:
            self.online_scal3r_manager.shutdown()
        
        return traj_full.inv().data.cpu().numpy()


def _online_context_frame_ids(frame_id):
    normalized = _to_int_or_none(frame_id)
    return [] if normalized is None else [normalized]


def _online_publication_control(video, current_frame_id, policy_current_frame_id):
    counter = int(video.counter.value)
    poll_frame_id = (
        _to_int_or_none(current_frame_id)
        if policy_current_frame_id is _USE_PUBLISHED_FRAME_ID
        else _to_int_or_none(policy_current_frame_id)
    )
    return counter, poll_frame_id


def _online_keyframe_snapshot(video, counter):
    indices = torch.arange(0, counter, device="cuda")
    poses_w2c = SE3(video.poses[indices]).matrix().detach().cpu()
    tstamps = video.tstamp[:counter].detach().cpu()
    intrinsics = video.intrinsics[:counter].detach().cpu() * 8.0
    images = video.images[torch.arange(0, counter)].detach().cpu()
    dba_depths = _online_dba_depths(video, counter)
    live_frame_ids = [int(tstamps[local_index].item()) for local_index in range(counter)]
    return poses_w2c, tstamps, intrinsics, images, dba_depths, live_frame_ids


def _online_dba_depths(video, counter):
    disps = video.disps_up[:counter].detach().cpu()
    valid = torch.isfinite(disps) & (disps > 0)
    return torch.where(valid, 1.0 / disps, torch.zeros_like(disps)).numpy()


def _to_int_or_none(value):
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def _frame_ids_for_viz_idx(video, viz_idx):
    if hasattr(viz_idx, "detach"):
        indices = viz_idx.detach().cpu().numpy().astype(int).tolist()
    else:
        indices = [int(item) for item in viz_idx]
    counter = int(video.counter.value)
    frame_ids = []
    for index in indices:
        if 0 <= index < counter:
            frame_ids.append(int(video.tstamp[index].item()))
    return frame_ids


def _online_scal3r_wait_summary(report):
    status_counts = _online_scal3r_status_counts(report)
    timing = _online_scal3r_timing_totals(report)
    parts = []
    if status_counts:
        parts.append(
            "states="
            + ",".join(f"{status}:{status_counts[status]}" for status in sorted(status_counts))
        )
    for label, value in (
        ("worker", timing.get("worker_wall")),
        ("runner", timing.get("runner_wall")),
        ("backend", timing.get("backend_wall")),
        ("align", timing.get("alignment_wall")),
    ):
        if value is not None:
            parts.append(f"{label}={value:.2f}s")
    return " ".join(parts)


def _online_scal3r_status_counts(report):
    alignment_results = report.get("alignment_results", []) or []
    worker_results = report.get("worker_results", []) or []
    alignment_chunk_ids = {
        str(item.get("chunk_id"))
        for item in alignment_results
        if item.get("chunk_id") is not None
    }
    terminal_items = list(alignment_results)
    terminal_items.extend(
        item
        for item in worker_results
        if str(item.get("chunk_id")) not in alignment_chunk_ids
    )
    counts = {}
    for item in terminal_items:
        status = item.get("status")
        if status is None:
            continue
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def _online_scal3r_timing_totals(report):
    totals = {}
    for item in (report.get("worker_results", []) or []):
        timing = item.get("timing_sec", {}) or {}
        for key in ("worker_wall", "runner_wall"):
            value = timing.get(key)
            if value is not None:
                totals[key] = totals.get(key, 0.0) + float(value)
        stages = item.get("stage_timings_sec", {}) or {}
        if isinstance(stages, dict):
            backend = stages.get("scal3r_backend", {}) or {}
            if isinstance(backend, dict):
                value = backend.get("model_postprocess_wall")
                if value is not None:
                    totals["backend_wall"] = totals.get("backend_wall", 0.0) + float(value)
    for item in (report.get("alignment_results", []) or []):
        timing = item.get("timing_sec", {}) or {}
        value = timing.get("alignment_wall")
        if value is not None:
            totals["alignment_wall"] = totals.get("alignment_wall", 0.0) + float(value)
    return totals


def _apply_online_scal3r_cli_overrides(config, args):
    if not getattr(args, "online_scal3r_block_until_ready", False) and (
        getattr(args, "online_scal3r_block_timeout_sec", None) is None
    ):
        return

    tracking = config.setdefault("Tracking", {})
    frontend = tracking.setdefault("frontend", {})
    online = frontend.setdefault("online_scal3r", {})
    if getattr(args, "online_scal3r_block_until_ready", False):
        online["block_until_prior_ready"] = True
    timeout = getattr(args, "online_scal3r_block_timeout_sec", None)
    if timeout is not None:
        online["block_until_prior_timeout_sec"] = float(timeout)
