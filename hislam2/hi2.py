import os
import torch
import numpy as np
from lietorch import SE3

from modules.droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from track_frontend import TrackFrontend
from track_backend import TrackBackend
from util.trajectory_filler import PoseTrajectoryFiller
from util.utils import load_config

from collections import OrderedDict
from torch.multiprocessing import Process, Queue
from gs_backend import GSBackEnd
from pgo_buffer import PGOBuffer


_USE_PUBLISHED_FRAME_ID = object()


class Hi2:
    def __init__(self, args):
        super(Hi2, self).__init__()
        self.load_weights(args.weights)
        self.config = config = load_config(args.config)
        self.args = args
        self.images = {}

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(config, args.image_size, args.buffer)
        frontend_config = config["Tracking"]["frontend"]
        self.online_scal3r_manager = self._build_online_scal3r_manager(config, args.output)
        self.scal3r_prior_provider = self._build_scal3r_prior_provider(frontend_config)
        self.video.use_scal3r_prior = self.scal3r_prior_provider is not None

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(
            self.net,
            self.video,
            config["Tracking"]["motion_filter"],
            prior_provider=self.scal3r_prior_provider,
        )

        # frontend process
        self.frontend = TrackFrontend(self.net, self.video, frontend_config)

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

    def _build_scal3r_prior_provider(self, frontend_config):
        if (
            self.online_scal3r_manager is not None
            and self.online_scal3r_manager.config.consume_frontend_prior
        ):
            print("[Online Scal3R] frontend depth-prior consumer enabled")
            return self.online_scal3r_manager.depth_prior_provider(consumer="frontend")

        scal3r_config = frontend_config.get("scal3r_prior", {})
        use_scal3r_prior = bool(frontend_config.get("use_scal3r_prior", scal3r_config.get("active", False)))
        if not use_scal3r_prior:
            return None
        from ffgs_slam.prior.hislam2_provider import Scal3RDepthPriorProvider

        provider = Scal3RDepthPriorProvider.from_config(scal3r_config)
        print("[Scal3R prior] enabled")
        return provider

    def _build_online_scal3r_manager(self, config, output_path):
        from ffgs_slam.online import OnlineScal3RManager

        return OnlineScal3RManager.from_config(config, output_path=output_path)

    def call_gs(self, viz_idx, dposes=None, dscale=None):
        data = {
            'viz_idx':  viz_idx.to(device='cpu'),
            'tstamp':   self.video.tstamp[viz_idx].to(device='cpu'),
            'poses':    self.video.poses[viz_idx].to(device='cpu'),
            'images':   self.video.images[viz_idx.cpu()],
            'normals':  self.video.normals[viz_idx.cpu()],
            'depths':   1./self.video.disps_up[viz_idx.cpu()],
            'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu') * 8,
            'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
            'scale_updates': dscale.to(device='cpu') if dscale is not None else None
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

        try:
            counter = int(self.video.counter.value)
            if counter <= 0:
                poll_frame_id = (
                    _to_int_or_none(current_frame_id)
                    if policy_current_frame_id is _USE_PUBLISHED_FRAME_ID
                    else _to_int_or_none(policy_current_frame_id)
                )
                manager.poll(current_frame_id=poll_frame_id)
                return

            indices = torch.arange(0, counter, device="cuda")
            poses_w2c = SE3(self.video.poses[indices]).matrix().detach().cpu()
            tstamps = self.video.tstamp[:counter].detach().cpu()
            intrinsics = (self.video.intrinsics[:counter].detach().cpu() * 8.0)
            images = self.video.images[torch.arange(0, counter)].detach().cpu()
            dba_depths = _online_dba_depths(self.video, counter)
            poll_frame_id = (
                _to_int_or_none(current_frame_id)
                if policy_current_frame_id is _USE_PUBLISHED_FRAME_ID
                else _to_int_or_none(policy_current_frame_id)
            )

            for local_index in range(counter):
                frame_id = int(tstamps[local_index].item())
                publish_policy_frame_id = (
                    frame_id
                    if policy_current_frame_id is _USE_PUBLISHED_FRAME_ID
                    else poll_frame_id
                )
                dba_depth = None if dba_depths is None else dba_depths[local_index]
                manager.publish_keyframe(
                    frame_id=frame_id,
                    keyframe_index=local_index,
                    image=images[local_index],
                    intrinsics=intrinsics[local_index],
                    pose_w2c=poses_w2c[local_index],
                    dba_depth=dba_depth,
                    metadata={"source": "hislam2_depth_video"},
                    policy_current_frame_id=publish_policy_frame_id,
                )
            manager.poll(current_frame_id=poll_frame_id)
        except Exception as exc:
            print(f"[Online Scal3R] publish/poll skipped: {exc}")

    def _drain_online_scal3r_before_final_refinement(self):
        manager = self.online_scal3r_manager
        if manager is None or manager.config.final_drain_timeout_sec <= 0:
            return

        current_frame_id = _last_online_frame_id(self.video, self.video.counter.value)
        print(
            "[Online Scal3R] final drain started "
            f"(timeout={manager.config.final_drain_timeout_sec:.1f}s)"
        )
        try:
            self._publish_online_scal3r_keyframes(
                current_frame_id=current_frame_id,
                policy_current_frame_id=None,
            )
            report = manager.drain(current_frame_id=None)
            print(
                "[Online Scal3R] final drain finished "
                f"submitted={len(report['submitted_chunks'])} "
                f"worker_results={len(report['worker_results'])} "
                f"alignment_results={len(report['alignment_results'])} "
                f"timed_out={report['timed_out']} "
                f"active_chunks={report.get('active_chunk_ids', [])} "
                f"inflight_chunks={report.get('inflight_chunk_ids', [])}"
            )
        except Exception as exc:
            print(f"[Online Scal3R] final drain skipped: {exc}")

    def terminate(self):
        """ terminate the visualization process, return poses [t, q] """
        self.video.ready.value = 1
        if self.pgba:
            dposes, dscale = self.video.pgobuf.run_pgba(self.LC_data_queue)
            if dposes is not None:
                self.call_gs(
                    torch.arange(0, self.video.counter.value, device='cuda'),
                    dposes, 
                    dscale
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
                depth, prior_conf = self.filterx.apply_external_prior(ind, depth)
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
        self._drain_online_scal3r_before_final_refinement()

        # final refinement
        self.call_gs(
            torch.arange(0, self.video.counter.value, device='cuda'), 
            dposes, 
            dscale
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


def _online_dba_depths(video, counter):
    try:
        disps = video.disps_up[:counter].detach().cpu()
        valid = torch.isfinite(disps) & (disps > 0)
        return torch.where(valid, 1.0 / disps, torch.zeros_like(disps)).numpy()
    except Exception:
        return None


def _to_int_or_none(value):
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def _last_online_frame_id(video, counter):
    try:
        count = int(counter)
        if count <= 0:
            return None
        return int(video.tstamp[count - 1].item())
    except Exception:
        return None
