import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
import torch
import cv2
import re
import os
import argparse
import time
import numpy as np
import lietorch
import resource
from queue import Empty
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
NUMERIC_TOKEN_RE = re.compile(r"[+]?(?:\d*\.\d+|\d+)")
READER_PAYLOAD_TIMEOUT_SEC = 60.0
READER_POLL_INTERVAL_SEC = 0.25
READER_JOIN_TIMEOUT_SEC = 5.0


class _DemoLifecycle:
    def __init__(self, queue, reader):
        self.queue = queue
        self.reader = reader
        self.hi2 = None
        self.progress = None
        self._reader_started = False
        self._reader_joined = False
        self._queue_closed = False
        self._progress_closed = False
        self._online_manager_shutdown = False

    def start_reader(self):
        self.reader.start()
        self._reader_started = True

    def attach_hi2(self, hi2):
        self.hi2 = hi2

    def attach_progress(self, progress):
        self.progress = progress

    def get_reader_payload(self):
        deadline = time.monotonic() + READER_PAYLOAD_TIMEOUT_SEC
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"image reader produced no payload within {READER_PAYLOAD_TIMEOUT_SEC:.1f} seconds"
                )
            try:
                return self.queue.get(timeout=min(READER_POLL_INTERVAL_SEC, remaining))
            except Empty:
                if not self.reader.is_alive():
                    raise RuntimeError(
                        "image reader exited before providing the next frame "
                        f"(exit code {self.reader.exitcode})"
                    )

    def join_reader(self, *, force=False):
        if not self._reader_started or self._reader_joined:
            return
        if force and self.reader.is_alive():
            self.reader.terminate()
        self.reader.join(timeout=READER_JOIN_TIMEOUT_SEC)
        if self.reader.is_alive():
            if not force:
                self.reader.terminate()
                self.reader.join(timeout=READER_JOIN_TIMEOUT_SEC)
            if self.reader.is_alive():
                kill = getattr(self.reader, "kill", None)
                if callable(kill):
                    kill()
                    self.reader.join(timeout=READER_JOIN_TIMEOUT_SEC)
        if self.reader.is_alive():
            raise RuntimeError("image reader did not exit during bounded cleanup")
        self._reader_joined = True

    def close_progress(self):
        if self.progress is None or self._progress_closed:
            return
        self._progress_closed = True
        self.progress.close()

    def close_queue(self):
        if self._queue_closed:
            return
        self._queue_closed = True
        cancel_join_thread = getattr(self.queue, "cancel_join_thread", None)
        if callable(cancel_join_thread):
            cancel_join_thread()
        self.queue.close()

    def mark_online_manager_shutdown(self):
        self._online_manager_shutdown = True

    def shutdown_online_manager(self):
        if self._online_manager_shutdown:
            return
        self._online_manager_shutdown = True
        manager = None if self.hi2 is None else getattr(self.hi2, "online_scal3r_manager", None)
        if manager is not None:
            manager.shutdown()

    def close(self, *, preserve_exception=False):
        cleanup_errors = []
        actions = (
            lambda: self.join_reader(force=True),
            self.shutdown_online_manager,
            self.close_progress,
            self.close_queue,
        )
        for action in actions:
            try:
                action()
            except Exception as exc:
                cleanup_errors.append(exc)
        if not cleanup_errors:
            return
        if preserve_exception:
            for exc in cleanup_errors:
                print(f"Demo cleanup failed: {exc}", file=sys.stderr)
            return
        raise cleanup_errors[0]


def extract_numeric_id(name):
    matches = NUMERIC_TOKEN_RE.findall(name)
    if not matches:
        return None
    return float(matches[-1])


def image_sort_key(name):
    numeric_id = extract_numeric_id(name)
    if numeric_id is None:
        return (1, name.lower(), name)
    return (0, numeric_id, name.lower(), name)


def list_image_files(imagedir):
    image_files = [
        name
        for name in os.listdir(imagedir)
        if os.path.isfile(os.path.join(imagedir, name)) and name.lower().endswith(IMAGE_SUFFIXES)
    ]
    return sorted(image_files, key=image_sort_key)


def select_image_files(imagedir, start=0, length=100000):
    image_files = list_image_files(imagedir)
    return image_files[start:start + length]


def image_timestamp(name, fallback):
    numeric_id = extract_numeric_id(name)
    return float(fallback) if numeric_id is None else numeric_id


def show_image(image, depth_prior, depth, normal):
    from util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()

    cv2.imshow(
        'rgb / prior normal / aligned prior depth / JDSA depth', 
        np.concatenate(
            (
                image / 255.0, 
                (normal[..., [2, 1, 0]] + 1.) / 2., 
                depth
            ), 
            axis=1
        )[::2,::2]
    )
    cv2.waitKey(1)


def mono_stream(queue, imagedir, calib, undistort=False, cropborder=False, start=0, length=100000, image_list=None):
    """ image generator """
    RES = 341 * 640

    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])

    if image_list is None:
        image_list = select_image_files(imagedir, start=start, length=length)

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intrinsics = calib[:4].astype(np.float32, copy=True)
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        image = np.ascontiguousarray(cv2.resize(image, (w1, h1)))

        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)

        is_last = (t == len(image_list)-1)
        queue.put((t, image, intrinsics.copy(), is_last))


def _queue_payload_to_tensors(image, intrinsics):
    """Rebuild Torch tensors in the consumer to avoid cross-process storage FDs."""

    if isinstance(image, torch.Tensor):
        image_tensor = image.detach().cpu()
        if image_tensor.ndim == 3:
            if image_tensor.shape[-1] == 3 and image_tensor.shape[0] != 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.ndim == 4:
            if image_tensor.shape[-1] == 3 and image_tensor.shape[1] != 3:
                image_tensor = image_tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"expected image payload with 3 or 4 dims, got {tuple(image_tensor.shape)}")
        image_tensor = image_tensor.contiguous()
    else:
        image_array = np.asarray(image)
        if image_array.ndim == 3:
            if image_array.shape[-1] == 3:
                image_tensor = torch.from_numpy(np.ascontiguousarray(image_array)).permute(2, 0, 1).unsqueeze(0)
            elif image_array.shape[0] == 3:
                image_tensor = torch.from_numpy(np.ascontiguousarray(image_array)).unsqueeze(0)
            else:
                raise ValueError(f"expected 3-channel image payload, got shape {image_array.shape}")
        elif image_array.ndim == 4:
            if image_array.shape[-1] == 3:
                image_tensor = torch.from_numpy(np.ascontiguousarray(image_array)).permute(0, 3, 1, 2)
            elif image_array.shape[1] == 3:
                image_tensor = torch.from_numpy(np.ascontiguousarray(image_array))
            else:
                raise ValueError(f"expected batched 3-channel image payload, got shape {image_array.shape}")
        else:
            raise ValueError(f"expected image payload with 3 or 4 dims, got shape {image_array.shape}")
        image_tensor = image_tensor.contiguous()

    intrinsics_tensor = torch.as_tensor(intrinsics, dtype=torch.float32).detach().cpu()
    if intrinsics_tensor.ndim == 1:
        intrinsics_tensor = intrinsics_tensor.unsqueeze(0)
    elif intrinsics_tensor.ndim != 2:
        raise ValueError(f"expected intrinsics payload with 1 or 2 dims, got {tuple(intrinsics_tensor.shape)}")
    return image_tensor, intrinsics_tensor.contiguous()


def save_trajectory(hi2, traj_full, imagedir, output, start=0, length=100000, image_list=None):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    if image_list is None:
        image_list = select_image_files(imagedir, start=start, length=length)
    tstamps_full = np.array([image_timestamp(name, index) for index, name in enumerate(image_list)])[..., np.newaxis]
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)  # for evo evaluation 
    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full)


def _count_nonempty_lines(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return None


def _format_fps_report(frames_processed, elapsed_sec, frame_source):
    frames = max(0, int(frames_processed))
    elapsed = max(0.0, float(elapsed_sec))
    minutes = elapsed / 60.0
    fps = frames / elapsed if elapsed > 0.0 else 0.0
    return (
        f"Frames processed: {frames} ({frame_source})\n"
        f"Main run elapsed time: {elapsed:.2f} s = {minutes:.2f} min\n"
        f"FPS: {frames} / {elapsed:.2f} = {fps:.2f} FPS\n"
    )


def write_fps_report(output, frames_processed, elapsed_sec):
    traj_full_path = os.path.join(output, "traj_full.txt")
    traj_full_count = _count_nonempty_lines(traj_full_path)
    if traj_full_count is not None:
        frames = traj_full_count
        frame_source = "traj_full.txt"
    else:
        frames = frames_processed
        frame_source = "tracking loop"

    fps_path = os.path.join(output, "fps.txt")
    with open(fps_path, "w", encoding="utf-8") as handle:
        handle.write(_format_fps_report(frames, elapsed_sec, frame_source))
    return fps_path


def save_dba_depths(hi2, output):
    depth_dir = os.path.join(output, "dba_depths")
    os.makedirs(depth_dir, exist_ok=True)

    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t].cpu().numpy().astype(int)
    disps = hi2.video.disps_up[:t].cpu().numpy().astype(np.float32)
    finite = np.isfinite(disps) & (disps > 0)
    depths = np.zeros_like(disps, dtype=np.float32)
    depths[finite] = 1.0 / disps[finite]

    for tstamp, depth in zip(tstamps, depths):
        np.save(os.path.join(depth_dir, f"{int(tstamp):06d}.npy"), depth.astype(np.float32, copy=False))
    print(f"Saved DBA depth npy files to: {depth_dir}")


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("expected TRUE or FALSE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")
    parser.add_argument(
        "--online_scal3r_block_until_ready",
        action="store_true",
        help="temporarily block before GS updates until online Scal3R priors are ready",
    )
    parser.add_argument(
        "--online_scal3r_block_timeout_sec",
        type=float,
        default=None,
        help="timeout for temporary online Scal3R prior blocking",
    )
    parser.add_argument(
        "--save_dba_depth",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="save final DBA/SLAM keyframe depths as .npy",
    )
    parser.add_argument(
        "--save_render_depth",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="save final GS-rendered keyframe depths as .npy",
    )

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')

    image_list = select_image_files(args.imagedir, start=args.start, length=args.length)
    if not image_list:
        raise ValueError(
            f"No input images selected from {args.imagedir} with --start {args.start} and --length {args.length}"
        )

    hi2 = None
    queue = Queue(maxsize=8)
    reader = Process(
        target=mono_stream, 
        args=(queue, args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length, image_list)
    )
    lifecycle = _DemoLifecycle(queue, reader)
    try:
        lifecycle.start_reader()

        N = len(image_list)
        args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
        pbar = tqdm(range(N), desc="Processing keyframes")
        lifecycle.attach_progress(pbar)
        frames_processed = 0
        tracking_started_at = None
        tracking_finished_at = None
        while 1:
            (t, image_payload, intrinsics_payload, is_last) = lifecycle.get_reader_payload()
            if tracking_started_at is None:
                tracking_started_at = time.time()
            image, intrinsics = _queue_payload_to_tensors(image_payload, intrinsics_payload)
            pbar.update()

            if hi2 is None:
                args.image_size = [image.shape[2], image.shape[3]]
                hi2 = Hi2(args)
                lifecycle.attach_hi2(hi2)

            hi2.track(
                t,
                image,
                intrinsics=intrinsics,
                is_last=is_last
            )

            if args.droidvis and hi2.video.tstamp[hi2.video.counter.value-1] == t:
                from geom.ba import get_prior_depth_aligned
                index = hi2.video.counter.value-2
                depth_prior, _ = get_prior_depth_aligned(
                    hi2.video.disps_prior_up[index][None].cuda(),
                    hi2.video.dscales[index][None]
                )
                show_image(
                    image[0],
                    1./depth_prior.squeeze(),
                    1./hi2.video.disps_up[index],
                    hi2.video.normals[index]
                )

            pbar.set_description(
                f"Processing keyframe No [{hi2.video.counter.value}] with GS num [{hi2.gs.gaussians._xyz.shape[0]}]"
            )
            frames_processed += 1
            tracking_finished_at = time.time()

            if is_last:
                lifecycle.close_progress()
                break

        lifecycle.join_reader()
        lifecycle.close_queue()
        traj = hi2.terminate()
        lifecycle.mark_online_manager_shutdown()
        if args.save_dba_depth:
            save_dba_depths(hi2, args.output)
        save_trajectory(
            hi2,
            traj,
            args.imagedir,
            args.output,
            start=args.start,
            length=args.length,
            image_list=image_list,
        )
        tracking_elapsed_sec = 0.0
        if tracking_started_at is not None and tracking_finished_at is not None:
            tracking_elapsed_sec = max(0.0, tracking_finished_at - tracking_started_at)
        write_fps_report(args.output, frames_processed, tracking_elapsed_sec)

        print("Done")
    finally:
        lifecycle.close(preserve_exception=sys.exc_info()[0] is not None)
