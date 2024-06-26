# We only support manually setting the bounding box of first frame and save the results in debug directory.

##########-------------- MixFormer-22k-----------------##########
#python tracking/video_demo.py mixformer_online baseline /YOUR/VIDEO/PATH  \
#   --optional_box [YOURS_X] [YOURS_Y] [YOURS_W] [YOURS_H] --params__model mixformer_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3

##########-------------- MixFormerL-22k-----------------##########
#python tracking/video_demo.py mixformer_online baseline /home/cyt/project/MixFormer/test.mp4  \
#   --optional_box 408 240 94 254 --params__model mixformerL_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3

# server 13
#python tracking/video_demo.py mixformer_online baseline /data0/cyt/experiments/mixformer/results_vis/v_4LXTUim5anY_c013.avi  \
#   --optional_box 509.0 318.0 72.0 175.0 --params__model mixformer_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3

#python tracking/video_demo.py mixformer_online baseline /data0/cyt/experiments/mixformer/results_vis/v_2ChiYdg5bxI_c120.avi  \
#  --optional_box 941.0 447.0 35.0 111.0 --params__model mixformer_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4.5 --params__update_interval 10 --params__online_sizes 3

#python tracking/video_demo.py mixformer_online baseline /data0/cyt/experiments/mixformer/results_vis/v_8rG1vjmJHr4_c004.avi  \
#  --optional_box 735.0 160.0 49.0 100.0 --params__model mixformer_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4 --params__update_interval 5 --params__online_sizes 5

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="7" \
python tracking/video_demo.py mixformer_vit_online baseline_large /media/data4/lizheng/MixFormer/UAV_demo_video/DJI_20230910153818_0003_S.MP4  \
  --params__model mixformer_vit_large_online.pth.tar --debug 1 \
  --params__search_area_scale 4.5 --params__update_interval 10 --params__online_sizes 5

# --optional_box 528 294 225 407

# DJI_20230910153025_0002_Z.MP4
# DJI_20230910153818_0003_Z.MP4

# DJI_20230910153050_0002_S.MP4
