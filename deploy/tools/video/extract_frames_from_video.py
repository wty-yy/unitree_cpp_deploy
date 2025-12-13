"""
从视频中按指定时间间隔提取帧并保存为图片存储到对应文件夹中。
python extract_frames_from_video.py \
    --video-file g1_dance_demo.mp4 \
    --output-folder g1_dance_demo_frames \
    --start 2 --end 4 --folder-duration 0.2
"""
import os
import cv2
import time
from pathlib import Path
from typing import Optional

def extract_frames(
        video_path: str,
        output_dir: Optional[str] = None,
        start: int = 0,
        end: Optional[int] = None,
        interval_sec: Optional[float] = None,
        folder_duration: Optional[float] = None,
        folder_prefix: str = "frame",
        folder_start_idx: int = 1,
    ):
    """
    从视频中按指定时间间隔提取帧并保存为图片存储到对应文件夹中, 例如:
        folder_duration=20, start=0, end=60, 则会创建3个文件夹:
        
        output_dir/
            1_frame0-19/
            2_frame20-39/
            3_frame40-59/
        
        每个文件夹内保存对应时间段的视频帧图片。

    Args:
        video_path (str): 输入视频文件的路径。
        output_dir (str): 保存图片的目标目录, 默认在video_path同级路径下创建同名文件夹。
        start (int): 开始时间点 (秒), 默认从0秒开始。
        end (Optional[int]): 结束时间点 (秒), 默认到视频结尾。
        interval_sec (Optional[float]): 提取帧的时间间隔 (秒), 默认提取所有帧。
        folder_duration (Optional[float]): 每个子文件夹包含的视频时长 (秒), 默认不分文件夹。
        folder_prefix (str): 子文件夹前缀名称, 默认 "frame"。
        folder_start_idx (int): 子文件夹起始索引, 默认从1开始。
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file {video_path} not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Can't open video {video_path}.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[ERROR] Can't get FPS.")
        cap.release()
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Process video: {video_path}")
    print(f"  - FPS: {fps:.2f}")

    if end is None or end > total_frames / fps:
        end = total_frames / fps
    if output_dir is None:
        output_dir = str(Path(video_path).parent / f"{Path(video_path).stem}_frames")
    output_dir = str(output_dir) + f"_{int(1/interval_sec) if interval_sec is not None else int(fps)}fps"
    output_dir = Path(output_dir)
    output_dirs = []
    output_dir.mkdir(parents=True, exist_ok=True)
    now_save_dir = output_dir / f"{folder_start_idx}_{folder_prefix}{int(start*fps)}-{int(end*fps-1)}"
    if folder_duration is None:
        now_save_dir.mkdir(parents=True, exist_ok=True)
        output_dirs.append(now_save_dir.name)
    now_end = None  # 当前子文件夹结束时间点 (秒)


    # 循环读取和保存帧
    frame_count = 0  # 记录总共读取了多少帧
    saved_count = 0  # 记录总共保存了多少帧
    next_save_time_sec = 0.0  # 下一个保存图片的目标时间点（秒）

    use_tqdm = True
    try:
        from tqdm import tqdm
        bar = tqdm(range(0, int(end * fps)), desc="Extracting frames", unit=" frames")
    except ImportError:
        bar = range(0, int(end * fps))
        use_tqdm = False

    start_time = time.time()
    for _ in bar:
        if cap.isOpened():
            flag, frame = cap.read()
        if not flag:
            break

        current_time_sec = frame_count / fps
        frame_count += 1
        if end is not None and current_time_sec >= end:
            break
        if current_time_sec < start:
            continue

        # 检查是否达到了保存时间点
        if current_time_sec > next_save_time_sec or interval_sec is None:
            if folder_duration is not None and (now_end is None or current_time_sec >= now_end):
                # 创建新的子文件夹
                now_save_dir = output_dir / f"{folder_start_idx}_{folder_prefix}{int(current_time_sec*fps)}-{int(min(current_time_sec+folder_duration, end)*fps-1)}"
                output_dirs.append(now_save_dir.name)
                now_save_dir.mkdir(parents=True, exist_ok=True)
                folder_start_idx += 1
                saved_count = 0
                now_end = current_time_sec + folder_duration

            output_filename = now_save_dir / f"{saved_count:05d}.jpg"
            cv2.imwrite(output_filename, frame)
            saved_count += 1
            if interval_sec is not None:
                next_save_time_sec += interval_sec
        if not use_tqdm:
            if frame_count % int(fps) == 0:
                time_used = time.time() - start_time
                print(f"  - Processed {frame_count}/{total_frames} frames, remaining {int(time_used)}<{int((total_frames - frame_count) / frame_count * time_used)} sec")

    cap.release()
    print(f"\n处理完成。")
    print(f"  - 总共读取帧数: {frame_count}")
    print(f"  - 图片保存至: '{output_dir}'")
    print(f"  - 包含子文件夹: {output_dirs}")

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--video-file", type=str, required=True, help="Path to the video file.")
parser.add_argument("--output-folder", type=str, help="Directory to save extracted frames.")
parser.add_argument("--start", type=int, default=0, help="Start time in seconds.")
parser.add_argument("--end", type=int, help="End time in seconds.")
parser.add_argument("--folder-duration", type=float, help="Duration of each folder in seconds.")
parser.add_argument("--interval-sec", type=float, help="Interval between extracted frames in seconds.")
parser.add_argument("--folder-prefix", type=str, default="frame", help="Prefix for folder names.")
parser.add_argument("--start-idx", type=int, default=1, help="Starting index for folder naming.")

args = parser.parse_args()

extract_frames(
    video_path=args.video_file,
    output_dir=args.output_folder,
    start=args.start,
    end=args.end,
    folder_duration=args.folder_duration,
    interval_sec=args.interval_sec,
    folder_prefix=args.folder_prefix,
    folder_start_idx=args.start_idx,
)
