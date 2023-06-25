PYTHONPATH="~/data4/MixFormer" \
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
python tracking/search_vis.py \
 --dataset_name VTUAV_TRAIN \
 --threads 24 \
 --tracker_name asymmetric_shared_online mixformer_vit_rgbt mixformer_vit_rgbt_unibackbone\
 --tracker_param attention_lasher_newfusion_2layer attention_lasher_newfusion_2layer attention_lasher_newfusion_2layer_2\
 --model \
 /home/lizheng/data4/MixFormer/results_train/checkpoints/train/asymmetric_shared_online/attention_lasher_newfusion_2layer/MixFormer_RGBT_OnlineScore_ep0020.pth.tar \
 /home/lizheng/data4/MixFormer/results_train/checkpoints/train/mixformer_vit_rgbt/attention_lasher_newfusion_2layer/MixFormer_RGBT_ep0070.pth.tar \
 /home/lizheng/data4/MixFormer/results_train/checkpoints/train/mixformer_vit_rgbt_unibackbone/attention_lasher_newfusion_2layer_2/MixFormer_RGBT_ep0055.pth.tar