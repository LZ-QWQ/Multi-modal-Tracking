# cd depthtrack_workspace
# OMP_NUM_THREADS=0 CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace ./ DepthTrack_ep05
# vot analysis --name DepthTrack_ep27
# cd ..

# cd depthtrack_workspace
# # 创建一个变量，包含所有的tracker名字
# trackers=""
# for epoch in {05,10,15}
# do
#     trackers+="DepthTrack_ep$epoch "
# done
# # 对所有的tracker执行评估
# nohup bash -c "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace ./ $trackers | tee /home/lizheng/data4/MixFormer/logs/depth_1.log" &
# cd ..
# trackers=""
# for epoch in {20,21,22}
# do
#     trackers+="DepthTrack_ep$epoch "
# done
# # 对所有的tracker执行评估
# nohup sh -c "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 vot evaluate --workspace ./ $trackers | tee /home/lizheng/data4/Mixformer/logs/depth_2.log" &

# trackers=""
# for epoch in {23,24,25,26}
# do
#     trackers+="DepthTrack_ep$epoch "
# done
# # 对所有的tracker执行评估
# nohup sh -c "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 vot evaluate --workspace ./ $trackers | tee /home/lizheng/data4/Mixformer/logs/depth_3.log" &

# trackers=""
# for epoch in {27,28,29,30}
# do
#     trackers+="DepthTrack_ep$epoch "
# done
# # 对所有的tracker执行评估
# nohup sh -c "OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 vot evaluate --workspace ./ $trackers | tee /home/lizheng/data4/Mixformer/logs/depth_4.log" &

# cd ..

cd depthtrack_workspace
# 创建一个变量，包含所有的tracker名字
trackers=""
for epoch in {5,10,15,20,21,22,23,24,25,26,27,28,29,30}
do
    trackers+="DepthTrack_ep$epoch "
done

# 对所有的tracker执行评估
# OMP_NUM_THREADS=0 CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace ./ $trackers

# 对所有的tracker执行分析
vot analysis --name $trackers
cd ..

