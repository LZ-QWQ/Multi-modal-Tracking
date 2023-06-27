import os
import glob

# LZ: 该文件根据测试的保存文件计算FPS（包含前后处理，但不包括cv2.imread），其实推理的时候应该做下预热的，但是平均下来FPS不低所以暂时不管了

# 跟踪器结果文件夹的路径
tracker_folder_path = "/home/lizheng/Multi-modal-Tracking/test/tracking_results/asymmetric_shared_online/tracking/RGBT234"

# 找到所有的***_time.txt文件
time_files = glob.glob(os.path.join(tracker_folder_path, '*_time.txt'))

# 用于存储所有帧的推理时间
all_inference_times = []

# 用于计数总帧数
total_frames = 0

for time_file in time_files:
    # 读取文件中的所有行
    with open(time_file, 'r') as f:
        lines = f.readlines()

    # 把每帧的推理时间加入列表
    all_inference_times.extend([float(line.strip()) for line in lines])

    # 更新总帧数
    total_frames += len(lines)

# 计算总的推理时间
total_inference_time = sum(all_inference_times)

# 计算平均FPS（每秒帧数）
avg_fps = total_frames / total_inference_time
print(f'Average FPS over all frames: {avg_fps}')
