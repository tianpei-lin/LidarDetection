#!/bin/bash

labels=(
20210906T144400_j7-00010_12_66to75.bag
)

bags_path="/home/tianpei.lin/data/bag/"
result_path="/home/tianpei.lin/data/prelabel/"

for bag in ${labels[*]}
do
if [ ! -f $bags_path/$bag ];then
    aws s3 cp s3://labeling/benchmark/obstacle_tracking/data/$bag ${bags_path} --endpoint-url=http://172.16.0.3 --profile sz
fi
done


# for label in ${labels[*]}
# do
# #if [ ! -f label/${label}.json ];then
#     aws s3 cp s3://labeling/benchmark/obstacle_tracking/label/${label}.json label/ --endpoint-url=http://172.16.0.3
# #fi
# done

cd /home/tianpei.lin/workspace/LidarDetection/pcdet/datasets/plusai

python /home/tianpei.lin/workspace/LidarDetection/tools/inference_bag2json.py \
  --cfg_file /home/tianpei.lin/workspace/LidarDetection/tools/cfgs/ouster_models/pv_rcnn_multiframe.yaml \
  --ckpt /home/tianpei.lin/workspace/LidarDetection/output/ouster_models/pv_rcnn_multiframe/default/ckpt/checkpoint_epoch_120.pth \
  --bag_file ${bags_path} \
  --save_path ${result_path}
