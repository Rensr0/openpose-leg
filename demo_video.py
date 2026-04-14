import numpy as np
import cv2
import os
import argparse
import time
from src import util
from src.body import Body

# 参数设置
parser = argparse.ArgumentParser(
        description="Process a video annotating leg poses detected.")
parser.add_argument('--file', type=str, required=True, help='Video file location to process.')
parser.add_argument('--skip', type=int, default=1,
    help='跳帧数：每N帧处理1帧（默认1，即全帧处理。设2则处理一半帧数）')
parser.add_argument('--scale', type=float, default=1.0,
    help='缩放比例：检测前缩小画面（默认1.0，建议0.5提速明显）')
parser.add_argument('--no-preview', action='store_true',
    help='关闭实时预览（大幅提速）')
parser.add_argument('--boxsize', type=int, default=368,
    help='模型输入尺寸（默认368，256更快但精度略降）')
args = parser.parse_args()

body_estimation = Body('model/body_pose_model.pth', boxsize=args.boxsize)

video_file = args.file
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_file}")
    exit(1)

input_fps = cap.get(cv2.CAP_PROP_FPS)
input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 输出文件名
base, ext = os.path.splitext(video_file)
output_file = base + '.processed.mp4'

writer = None
frame_count = 0
processed_count = 0
start_time = time.time()

print(f"视频: {video_file} ({input_width}x{input_height}, {input_fps:.1f}fps, 共{total_frames}帧")
print(f"设置: 跳帧={args.skip}, 缩放={args.scale}, boxsize={args.boxsize}, 预览={'关' if args.no_preview else '开'}")

# 预览窗口：可缩放，自动适配屏幕
if not args.no_preview:
    cv2.namedWindow('frame - legs only', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # 取屏幕 85% 作为最大预览尺寸
    try:
        # 尝试获取屏幕分辨率（不同平台方法不同）
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
    except Exception:
        sw, sh = 1920, 1080  # 保守默认
    max_w = int(sw * 0.85)
    max_h = int(sh * 0.85)
    # 等比缩放到屏幕内
    ratio = min(max_w / input_width, max_h / input_height, 1.0)
    win_w = int(input_width * ratio)
    win_h = int(input_height * ratio)
    cv2.resizeWindow('frame - legs only', win_w, win_h)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 跳帧
    if (frame_count - 1) % args.skip != 0:
        continue

    # 缩放
    if args.scale != 1.0:
        small = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    else:
        small = frame

    # 推理
    candidate, subset = body_estimation(small)

    # 关键点坐标还原到原始尺寸
    if args.scale != 1.0 and len(candidate) > 0:
        candidate[:, 0] /= args.scale
        candidate[:, 1] /= args.scale

    canvas = frame.copy()
    canvas = util.draw_bodypose(canvas, candidate, subset)
    processed_count += 1

    if writer is None:
        writer = cv2.VideoWriter(output_file, fourcc, input_fps, (canvas.shape[1], canvas.shape[0]))

    writer.write(canvas)

    # 稀疏进度输出（每 30 帧一次）
    if processed_count % 30 == 0:
        elapsed = time.time() - start_time
        fps = processed_count / elapsed
        eta = (total_frames // args.skip - processed_count) / fps if fps > 0 else 0
        print(f"进度: {processed_count}/{total_frames // args.skip} | "
              f"速度: {fps:.1f}fps | 剩余: {eta:.0f}s")

    if not args.no_preview:
        cv2.imshow('frame - legs only', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_time
print(f"\n处理完成！")
print(f"  总帧数: {frame_count}，实际检测: {processed_count}帧")
print(f"  耗时: {elapsed:.1f}s，平均速度: {processed_count/elapsed:.1f}fps")
print(f"  输出: {output_file}")
