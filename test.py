from concurrent import futures
import torch
import multiprocessing as mp
import time
import os

# def init_worker(args):
#     idx = mp.current_process()._identity[0]  # from 1
#     pid = mp.current_process().pid
#     print("{} process initialize".format(pid))
#     args.data =idx * 10
#     print("args.data", args.data, id(args))

# def print_(test_, idx):
#      print(test_.data, mp.current_process().pid, id(test_))
#      time.sleep(0.5)

# def main(test_1):

#     print(id(test_1))
#     with futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"), initializer=init_worker, initargs=(test_1,)) as executor:
#             fs = [executor.submit(print_, test_1, idx) for idx in range(5)]
#             for _, f in enumerate(futures.as_completed(fs)):
#                 if f.exception() is not None and f.exception() != 0:
#                     print("[Error]", f.exception(), ', for the detail using "try f.results() + except"')
#                     print(f.result())
#             print("{} videos done!".format(len(fs)))

# if __name__ == "__main__":
#     main()
# import cv2
# import os
# import numpy as np
# save_dir = "./visualization/train_data"
# os.makedirs(save_dir, exist_ok=True)

# cv2.imwrite(os.path.join(save_dir, "test.png"), 255*np.ones([255,255,3]))


# import numpy as np
# import pandas
# res = pandas.read_csv(
#             "test.txt", delimiter=",", header=None, dtype=np.float32, na_filter=True, low_memory=False
#         ).values
# print(res, res>0, res < 1)


# import numpy as np
# import cv2

# im_i = cv2.imread("/home/lizheng/data4/MixFormer/data/VTUAV/train_data/train_LT_001/bus_005/ir/000001.jpg")

# im_i_color = cv2.applyColorMap(im_i, cv2.COLORMAP_JET)

# cv2.imwrite("test.png", np.concatenate([im_i, im_i_color], axis=1))

import cv2

def test_image_reading(image_folder):
    # 获取文件夹中所有图片的文件名
    image_files = os.listdir(image_folder)

    # 逐个读取图片并计算读取时间
    total_time = 0
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        start_time = time.time()
        image = cv2.imread(image_path)  # 使用OpenCV读取图片
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

    # 计算平均读取时间
    num_images = len(image_files)
    average_time = total_time / num_images
    print(f"Average image reading time: {average_time} seconds")

# 指定存储图片的文件夹路径
image_folder = "/home/lizheng/data4/MixFormer/data/VTUAV/train_data/train_LT_001/bus_005/rgb"

# 运行测试
test_image_reading(image_folder)