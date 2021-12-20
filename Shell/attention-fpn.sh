time=$(date "+%Y%m%d")
work_folder=/home/user/sun_chen/Work_dir/AttentionFPN/${time}

for attention in attention-fpn context-r4-fpn context-r16-fpn nonlocal-fpn;

do
config_file=./Model/Model_Fewshot/attention_base/tfa_101_fpn_coco_base_$attention.py

python ./third_party/mmfewshot/tools/detection/train.py \
    $config_file \
    --work-dir ${work_folder}/$attention \
    --gpu-ids 1

# CUDA_VISIBLE_DEVICES=0,1 bash ./third_party/mmfewshot/tools/detection/dist_train.sh \
#     $config_file \
#     2
#     --work-dir ${work_folder}/$attention 

done
