# Different test settings for MixFormer-ViT-b, MixFormer-ViT-l on LaSOT/TrackingNet/GOT10K/UAV123/OTB100
# First, put your trained MixFomrer-online models on SAVE_DIR/models directory. 
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- MixViT-B -----------------##########
# 记得改 tracking.yaml 里面的 LOAD_FROME_TRAIN_RESULT 谁写的这个代码QAQ !!!

# 直接测试
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7" \
# python tracking/test.py mixformer_vit_online baseline --dataset VTUAV \
#  --threads 24 --num_gpus 4 --params__model mixformer_vit_base_online.pth.tar \
#  --type RGB
#  --save_name_suffix RGB

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python tracking/test.py asymmetric_shared attention_lasher_newfusion_2layer_vtuav_normal --dataset VTUAV \
 --threads 16 --num_gpus 4 --params__model VTUAV_SOTA.tar \
 --type RGBT --save_name_suffix jet_45_VTUAV_SOTA_no_update

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5" \
# python tracking/test.py asymmetric_shared attention_lasher_newfusion_2layer --dataset LasHeR \
#  --threads 6 --params__model MixFormer_RGBT_ep0095.pth.tar \
#  --type RGBT --save_name_suffix jet_45_lasher_SOTA_no_update

# ===========================================================================================

# 训练结果测试

# LZ for test
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3" \
# python tracking/test.py asymmetric_shared attention_lasher_newfusion_2layer_vtuav_normal --dataset VTUAV \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7" \
# python tracking/test.py mixformer_vit baseline_large_tir --dataset RGBT234 \
#  --threads 24 --checkpoint_dir ./results_train \
#  --type TIR  --save_name_suffix TIR_fintune

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0, 1, 2, 3, 4, 5, 6, 7" \
# python tracking/test.py mixformer_vit_rgbt_unibackbone attention_lasher_newfusion_2layer_2 --dataset RGBT234 \
#  --threads 24 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0, 1, 2, 3" \
# python tracking/test.py mixformer_vit_rgbt_unibackbone attention_lasher_newfusion_2layer --dataset LasHeR \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python tracking/test.py asymmetric_shared attention_lasher_newfusion_2layer_RGBD --dataset DepthTrack \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python tracking/test.py asymmetric_shared attention_lasher_newfusion_2layer_3 --dataset RGBT234 \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python tracking/test.py asymmetric_shared attention_lasher_newfusion_2layer_3 --dataset LasHeR \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45



# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3" \
# python tracking/test.py asymmetric_shared_ce attention_lasher_newfusion_2layer --dataset RGBT234 \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3" \
# python tracking/test.py asymmetric_shared_ce attention_lasher_newfusion_2layer --dataset LasHeR \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3" \
# python tracking/test.py asymmetric_shared_online attention_lasher_newfusion_2layer_vtuav_normal_load --dataset VTUAV \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python tracking/test.py asymmetric_shared_online attention_lasher_newfusion_2layer --dataset LasHeR \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python tracking/test.py asymmetric_shared_online attention_lasher_newfusion_2layer_RGBD_load --dataset DepthTrack \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python tracking/test.py asymmetric_shared_online attention_lasher_newfusion_2layer_load --dataset LasHeR \
#  --threads 12 --checkpoint_dir ./results_train \
#  --type RGBT  --save_name_suffix jet_45


# --params__search_area_scale 5.0


# python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

### TrackingNet test and pack
# python tracking/test.py mixformer_vit_online baseline --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online.pth.tar
# python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_vit_online --cfg_name baseline

### GOT10k test and pack
# python tracking/test.py mixformer_vit_online baseline --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online_got.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name mixformer_vit_online --cfg_name baseline

### UAV123
# python tracking/test.py mixformer_vit_online baseline --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online.pth.tar --params__search_area_scale 4.7
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline

### OTB100
#python tracking/test.py mixformer_cvt_online baseline --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_vit_base_online_22k.pth.tar --params__search_area_scale 4.45
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline


##########-------------- MixViT-L -----------------##########
### LaSOT test and evaluation
# python tracking/test.py mixformer_vit_online baseline_large --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online.pth.tar --params__search_area_scale 4.55
# python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large

### TrackingNet test and pack
# python tracking/test.py mixformer_vit_online baseline_large --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online.pth.tar --params__search_area_scale 4.6
# python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_vit_online --cfg_name baseline_large

### GOT10k test and pack
# python tracking/test.py mixformer_vit_online baseline_large --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online_got.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name mixformer_vit_online --cfg_name baseline_large

### UAV123
# python tracking/test.py mixformer_vit_online baseline_large --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online.pth.tar --params__search_area_scale 4.7
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_large

### OTB100
#python tracking/test.py mixformer_cvt_online baseline_large --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_vit_large_online_22k.pth.tar --params__search_area_scale 4.6
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_large


