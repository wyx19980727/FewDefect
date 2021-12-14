#############################################
############  train & fine-tune #############
#############################################

### base class train
time=$(date "+%Y%m%d")
# work_folder=/home/dlsuncheng/Work_dir/FsMMdet/${time}

work_folder=/home/dlsuncheng/Work_dir/NWPU/${time}


# CUDA_VISIBLE_DEVICES=1,2 bash ./MMFewShot/tools/detection/dist_train.sh \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py \
#     --work-dir ${work_folder}"/base_train" \
#     2

# python ./MMFewShot/tools/detection/train.py \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_base-training.py \
#     --work-dir ${work_folder}"/base_train_10000iter" \
#     --gpu-ids 1

python ./MMFewShot/tools/detection/train.py \
    /home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/meta_rcnn/nwpuv2/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py \
    --work-dir ${work_folder}"/NWPU" \
    --gpu-ids 1

# step2: reshape the bbox head of base model for few shot fine-tuning
# base model needs to be initialized with following script: tools/detection/misc/initialize_bbox_head.py

# python -m tools.detection.misc.initialize_bbox_head \
#     --src1 ${work_folder}"/base_train/best_bbox_mAP.pth"
#     --method randinit \
#     --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training

# step3: few shot fine-tuning
# python ./MMFewShot/tools/detection/train.py \

bash ./MMFewShot/tools/detection/dist_train.sh \
    /home/dlsuncheng/Projects/FSOD/FsMMdet/Model_Fewshot/tfa/neu_det/tfa_r101_fpn_coco_10shot-fine-tuning_less.py \
    4 \
    --work-dir ${work_folder}"/fine-tune_less_4GPU" \
   
### novel class few shot fine-tune

### combine weights


#############################################
############   test & evaluate  #############
#############################################