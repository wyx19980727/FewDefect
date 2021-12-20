#############################################
############  train & fine-tune #############
#############################################

### base class train
time=$(date "+%Y%m%d")
work_folder=/home/user/sun_chen/Work_dir/AttentionFPN/${time}

# CUDA_VISIBLE_DEVICES=1,2 bash ./MMFewShot/tools/detection/dist_train.sh \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py \
#     --work-dir ${work_folder}"/base_train" \
#     2
for i in
do
    python ./third_party/mmfewshot/tools/detection/train.py \
        /home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py \
        --work-dir ${work_folder}"/base_train_10000iter" \
        --gpu-ids 1
done

# step2: reshape the bbox head of base model for few shot fine-tuning
# base model needs to be initialized with following script: tools/detection/misc/initialize_bbox_head.py

# python -m tools.detection.misc.initialize_bbox_head \
#     --src1 ${work_folder}"/base_train/best_bbox_mAP.pth"
#     --method randinit \
#     --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training

# step3: few shot fine-tuning
# python ./MMFewShot/tools/detection/train.py \

# bash ./third_party/mmfewshot/tools/detection/dist_train.sh \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning_less.py \
#     4 \
#     --work-dir ${work_folder}"/fine-tune_less_4GPU" \
   

#############################################
############   test & evaluate  #############
#############################################
# python ./third_party/mmfewshot/tools/detection/test.py \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning_less.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211203/fine-tune_less_4GPU/best_bbox_mAP.pth \
#     --eval bbox 

