import cv2
import subprocess
import os


def video_cut_speedup(video_path, output_path, start_frame, end_frame):
    # 加载视频
    video = cv2.VideoCapture(video_path)

    # 获取视频的基本信息
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # 创建视频写入对象，设置帧率为原来的两倍，达到加速效果
    out = cv2.VideoWriter(output_path, fourcc, fps * 2.5, (width, height))

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 裁切指定的帧范围
        if start_frame <= frame_count <= end_frame:
            out.write(frame)

        frame_count += 1

    video.release()
    out.release()


# 调用函数
video_cut_speedup("/home/lizheng/data4/MixFormer/test/result_plots/VTUAV/vis_videos/car_132.mp4", "temp.mp4", 15 * 30, 50 * 30)
subprocess.run(
    args=[
        "/usr/bin/ffmpeg",
        "-nostdin",  # https://github.com/kkroening/ffmpeg-python/issues/108 解决回显关闭问题
        "-y",
        "-loglevel",
        "quiet",
        "-i",
        "temp.mp4",
        "-vcodec",
        "h264",
        os.path.join("output.mp4"),
    ],
    check=True,
)
os.remove("temp.mp4")