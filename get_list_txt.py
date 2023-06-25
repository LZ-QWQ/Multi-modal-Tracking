import os

# 指定要列出子文件夹的父文件夹
parent_folder = "/home/lizheng/data4/MixFormer/depthtrack_workspace/sequences"

# 用于保存子文件夹名的列表
folder_names = []

# os.walk 会遍历给定目录下的所有文件和文件夹，包括子目录
for path, dirs, files in os.walk(parent_folder):
    # 仅在父目录级别添加子文件夹名
    if path == parent_folder:
        folder_names.extend(dirs)

# 将子文件夹名逐行写入 list.txt 文件
with open("list.txt", "w") as f:
    for folder in folder_names:
        f.write(folder + "\n")
