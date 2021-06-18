#!/bin/bash

labels=(
20210502T063909_j7-l4e-00011_7_194to214.bag
20210502T063909_j7-l4e-00011_7_228to248.bag
20210502T111802_j7-l4e-00011_11_254to274.bag
20210502T111802_j7-l4e-00011_13_2to22.bag
20210502T111802_j7-l4e-00011_27_117to137.bag
20210502T111802_j7-l4e-00011_27_141to161.bag
20210502T111802_j7-l4e-00011_28_100to120.bag
20210502T111802_j7-l4e-00011_44_119to139.bag
20210502T111802_j7-l4e-00011_44_173to193.bag
20210502T111802_j7-l4e-00011_58_101to121.bag
20210506T054852_j7-l4e-00011_10_94to114.bag
20210506T054852_j7-l4e-00011_1_218to238.bag
20210506T054852_j7-l4e-00011_23_186to206.bag
20210506T054852_j7-l4e-00011_24_130to140.bag
20210506T054852_j7-l4e-00011_24_215to235.bag
20210506T054852_j7-l4e-00011_25_14to34.bag
20210506T054852_j7-l4e-00011_36_137to157.bag
20210506T054852_j7-l4e-00011_55_107to127.bag
20210506T054852_j7-l4e-00011_55_51to71.bag
20210506T054852_j7-l4e-00011_56_45to65.bag
20210506T054852_j7-l4e-00011_57_51to71.bag
20210506T054852_j7-l4e-00011_58_47to67.bag
20210506T111749_j7-l4e-00011_10_191to211.bag
20210506T111749_j7-l4e-00011_1_18to38.bag
20210506T111749_j7-l4e-00011_1_68to88.bag
20210506T111749_j7-l4e-00011_25_267to287.bag
20210506T111749_j7-l4e-00011_27_124to144.bag
20210506T111749_j7-l4e-00011_27_44to64.bag
20210506T111749_j7-l4e-00011_28_23to43.bag
20210506T111749_j7-l4e-00011_32_33to53.bag
20210506T111749_j7-l4e-00011_32_73to93.bag
20210506T111749_j7-l4e-00011_32_8to28.bag
20210506T111749_j7-l4e-00011_37_269to289.bag
20210506T111749_j7-l4e-00011_40_115to135.bag
20210506T111749_j7-l4e-00011_41_18to38.bag
20210506T111749_j7-l4e-00011_44_139to159.bag
20210506T111749_j7-l4e-00011_45_149to169.bag
20210506T111749_j7-l4e-00011_45_79to99.bag
20210506T111749_j7-l4e-00011_47_108to128.bag
20210506T111749_j7-l4e-00011_47_1to21.bag
20210506T111749_j7-l4e-00011_47_38to58.bag
20210506T111749_j7-l4e-00011_47_68to88.bag
20210506T111749_j7-l4e-00011_5_18to38.bag
20210506T111749_j7-l4e-00011_5_247to267.bag
20210506T111749_j7-l4e-00011_6_71to91.bag
20210508T083602_j7-l4e-00011_11_239to259.bag
20210508T102051_j7-l4e-00011_12_259to279.bag
20210508T102051_j7-l4e-00011_13_14to34.bag
20210508T102051_j7-l4e-00011_18_136to166.bag
20210508T102051_j7-l4e-00011_23_235to255.bag
20210508T102051_j7-l4e-00011_27_202to222.bag
20210508T102051_j7-l4e-00011_27_260to280.bag
20210508T102051_j7-l4e-00011_29_227to247.bag
20210508T102051_j7-l4e-00011_9_73to93.bag
20210508T132605_j7-l4e-00011_15_253to273.bag
20210508T132605_j7-l4e-00011_43_53to73.bag
20210509T085839_j7-l4e-00011_28_136to156.bag
20210509T085839_j7-l4e-00011_34_101to121.bag
20210509T085839_j7-l4e-00011_35_257to277.bag
20210509T085839_j7-l4e-00011_36_6to26.bag
20210509T085839_j7-l4e-00011_37_88to108.bag
20210509T123652_j7-l4e-00011_29_201to221.bag
20210509T123652_j7-l4e-00011_5_141to161.bag
20210509T123652_j7-l4e-00011_5_173to193.bag
20210509T151749_j7-l4e-00011_0_181to201.bag
20210509T151749_j7-l4e-00011_39_259to279.bag
20210509T151749_j7-l4e-00011_42_121to141.bag
20210509T151749_j7-l4e-00011_43_75to95.bag
20210509T151749_j7-l4e-00011_44_261to281.bag
20210509T151749_j7-l4e-00011_46_176to196.bag
20210509T151749_j7-l4e-00011_46_259to279.bag
20210509T151749_j7-l4e-00011_46_89to109.bag
20210509T151749_j7-l4e-00011_47_227to247.bag
20210509T151749_j7-l4e-00011_47_257to277.bag
20210509T151749_j7-l4e-00011_47_2to22.bag
20210509T151749_j7-l4e-00011_47_77to97.bag
20210509T151749_j7-l4e-00011_6_253to273.bag
)

bags_path="/old_home/archive/tianpei/data/bag/"
result_path="/old_home/archive/tianpei/data/prelabel/"

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
  --cfg_file cfgs/livox_models/pv_rcnn_multiframe.yaml \
  --ckpt ../checkpoints/livox_models/pv_rcnn_multiframe/checkpoint_epoch_80.pth \
  --bag_file ${bags_path} \
  --save_path ${result_path}
