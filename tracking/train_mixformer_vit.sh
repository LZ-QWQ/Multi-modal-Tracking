# There are the detailed training settings for MixFormer-ViT-B and MixFormer-ViT-L.
# 1. download pretrained ViT-MAE models (mae_pretrain_vit_base.pth.pth/mae_pretrain_vit_large.pth) at https://github.com/facebookresearch/mae
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer_vit/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training MixFormer-ViT-B
# Stage1: train mixformer without SPM
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python ./tracking/train.py --script asymmetric_shared \
#  --config attention_lasher_newfusion_2layer_RGBD --save_dir ./results_train --mode multiple --nproc_per_node 4



# two-stream
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python ./tracking/train.py --script mixformer_vit_rgbt \
# --config attention_lasher_newfusion_2layer --save_dir ./results_train --mode multiple --nproc_per_node 4

# # single-stream
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="0,1,2,3" \
# python ./tracking/train.py --script mixformer_vit_rgbt_unibackbone \
#  --config attention_lasher_newfusion_2layer_2 --save_dir ./results_train --mode multiple --nproc_per_node 4

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python ./tracking/train.py --script asymmetric_shared \
#  --config attention_lasher_newfusion_2layer_3 --save_dir ./results_train --mode multiple --nproc_per_node 4

# Stage2: train mixformer_online, i.e., SPM (score prediction module)
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="4,5,6,7" \
# python ./tracking/train.py --script asymmetric_shared_online \
#  --config attention_lasher_newfusion_2layer_RGBD_load --save_dir ./results_train --mode multiple --nproc_per_node 4

### Training MixFormer-L
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="0,1,2,3" \
python ./tracking/train.py --script mixformer_vit \
 --config baseline_large_tir --save_dir ./results_train --mode multiple --nproc_per_node 4

#python tracking/train.py --script mixformer_vit --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-B_GOT
#python tracking/train.py --script mixformer_vit --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_vit_online --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT_ONLINE --mode multiple --nproc_per_node 8 \
#    --stage1_model /STAGE1/MODEL
