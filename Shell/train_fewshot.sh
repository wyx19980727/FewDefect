#############################################
############  train & fine-tune #############
#############################################
time=20211130
work_folder=/home/dlsuncheng/Work_dir/FsMMdet/${time}
### base class train
python ./MMdet/tools/train.py \
    /home/dlsuncheng/Projects/FSOD/FsMMdet/Model/base_pretrain/frcn_fc_base_pretrain.py \
    --work-dir ${work_folder}"/base_train_epochrunner" \
    --gpu-ids 2
### base class test
# python ./MMdet/tools/test.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211101/20211101/fc_base_pretrain/frcn_fc_base_pretrain.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211101/20211101/fc_base_pretrain/epoch_24.pth \
#     --eval bbox \

### novel class few shot fine-tune
# time=$(date "+%Y%m%d")
# train_folder=/home/dlsuncheng/Projects/FSOD/FsMMdet/Model/few_shot/
# work_folder=/home/dlsuncheng/Work_dir/FsMMdet/${time}
# gpu_ids=1

# for i in 3 5 10 30;
# do
# novel_train_file=${train_folder}"novel/fc_cos-ft/frcn_cos_unfreeze_novel_"$i"shot.py"
# all_train_file=${train_folder}"all/fc_cos-ft/frcn_cos_unfreeze_all_"$i"shot.py"

# work_dir=${work_folder}"/FewShot/"${i}shot

# ### train
# python ./MMdet/tools/train.py \
#     ${novel_train_file} \
#     --work-dir ${work_dir}"/novel" \
#     --gpu-ids ${gpu_ids} 

# ### combine weights
# python ./Tools/pth_surgery.py \
#     --pth1_path ${work_dir}/novel/best_bbox_mAP.pth \
#     --pth2_path ./Weights/base_best_bbox_mAP.pth \
#     --save_dir ./Weights \
#     --method "combine" \
#     --tar-name  "model_reset_"$i"shot_2"

# weights_save_path="./Weights/model_reset_"$i"shot_2_combine.pth"

# ### fine tune on all dataset
# python ./MMdet/tools/train.py \
#     ${all_train_file} \
#     --work-dir ${work_dir}"/all" \
#     --gpu-ids ${gpu_ids} \
#     --options "load_from="${weights_save_path}

#############################################
############   test & evaluate  #############
#############################################

# python ./MMdet/tools/test.py \
#     ${all_train_file} \
#     ${work_dir}/all/best_bbox_mAP.pth \
#     --eval bbox \
#     --options "iou_thrs=[0.5]"


# done 





