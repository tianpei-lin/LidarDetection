#!/bin/bash

labels=(
20210508T162630_paccar-k002dc_13_0to20.db
20210416T135625_paccar-k002dc_2_40to60.db
20210416T130638_paccar-k002dc_4_200to220.db
)

bags_path="/home/tianpei.lin/data/train/bag/"
result_path="/home/tianpei.lin/data/train/prelabel/"

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

python inference_bag2json.py \
  --cfg_file cfgs/ouster_models/pv_rcnn_multiframe.yaml \
  --ckpt /home/tianpei.lin/checkpoints/ouster_models/pv_rcnn_multiframe/checkpoint_epoch_80.pth \
  --bag_file ${bags_path} \
  --save_path ${result_path}
