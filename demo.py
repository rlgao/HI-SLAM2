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
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2


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


def mono_stream(queue, imagedir, calib, undistort=False, cropborder=False, start=0, length=100000):
    """ image generator """
    RES = 341 * 640

    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])

    image_list = sorted(os.listdir(imagedir))[start:start+length]

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


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    tstamps_full = np.array([float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1]) for x in sorted(os.listdir(imagedir))[start:]])[..., np.newaxis]
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

    hi2 = None
    queue = Queue(maxsize=8)
    reader = Process(
        target=mono_stream, 
        args=(queue, args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length)
    )
    reader.start()

    N = len(os.listdir(args.imagedir))
    args.buffer = min(1000, N // 10 + 150) if args.buffer < 0 else args.buffer
    pbar = tqdm(range(N), desc="Processing keyframes")
    frames_processed = 0
    tracking_started_at = None
    tracking_finished_at = None
    while 1:
        (t, image_payload, intrinsics_payload, is_last) = queue.get()
        if tracking_started_at is None:
            tracking_started_at = time.time()
        image, intrinsics = _queue_payload_to_tensors(image_payload, intrinsics_payload)
        pbar.update()

        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)

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
            pbar.close()
            break

    reader.join()

    try:
        traj = hi2.terminate()
    finally:
        online_manager = None if hi2 is None else getattr(hi2, "online_scal3r_manager", None)
        if online_manager is not None:
            online_manager.shutdown()
    if args.save_dba_depth:
        save_dba_depths(hi2, args.output)
    save_trajectory(
        hi2, 
        traj, 
        args.imagedir, 
        args.output, 
        start=args.start
    )
    tracking_elapsed_sec = 0.0
    if tracking_started_at is not None and tracking_finished_at is not None:
        tracking_elapsed_sec = max(0.0, tracking_finished_at - tracking_started_at)
    write_fps_report(args.output, frames_processed, tracking_elapsed_sec)

    print("Done")
