#!/bin/bash

labels=(
20210414T181240_paccar-k002dc_12_0to22.bag
20210414T181240_paccar-k002dc_12_37to48.bag
20210414T181240_paccar-k002dc_1_50to81.bag
20210414T181240_paccar-k002dc_17_37to53.bag
20210414T181240_paccar-k002dc_17_4to31.bag
20210414T181240_paccar-k002dc_19_73to92.bag
20210414T181240_paccar-k002dc_19_95to113.bag
20210414T181240_paccar-k002dc_2_40to51.bag
20210414T181240_paccar-k002dc_3_0to33.bag
20210414T181240_paccar-k002dc_4_100to113.bag
20210414T181240_paccar-k002dc_5_0to32.bag
20210414T181240_paccar-k002dc_8_79to111.bag
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

python inference_bag2json.py \
  --cfg_file cfgs/ouster_models/pv_rcnn_multiframe.yaml \
  --ckpt /home/tianpei.lin/workspace/LidarDetection/output/ouster_models/pv_rcnn_multiframe/default/ckpt/checkpoint_epoch_80.pth \
  --bag_file ${bags_path} \
  --save_path ${result_path}
