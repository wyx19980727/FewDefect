#############################################
##  baseline
#############################################

# # baseline 

# python ./MMdet/tools/test.py \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model/full_data/frcn_all.py \
#     /home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_23.pth \
#     --eval bbox 
    # --eval mAP \

#     # --options "iou_thrs=[0.5]"    

# python ./MMdet/tools/test.py \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model/full_data/frcn_all.py \
#     /home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_23.pth \
#     --eval bbox \
#     --eval-options "iou_thrs=[0.5]" 



# # zero-dce 
# python ./MMdet/tools/test.py \
#     /home/dlsuncheng/Projects/FSOD/FsMMdet/Model/full_data/augment/frcn_all_zero_dce.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211102/FRCN_all/zero_dce/best_bbox_mAP.pth \
#     --eval bbox \
#     --options "iou_thrs=[0.5]"    

#############################################
##  self-attention
#############################################

# time=$(date "+%Y%m%d")
time=20211111
train_folder=/home/dlsuncheng/Projects/FSOD/FsMMdet/Model
work_folder=/home/dlsuncheng/Work_dir/FsMMdet/${time}

for i in frcn_all_gc_r16 #frcn_all_gc_r16 frcn_all_non_local_fpn;  

do  
train_file=${train_folder}"/full_data/self-attention/"$i".py"
work_dir=${work_folder}"/softnms/"$i

# #### train
# # python ./MMdet/tools/train.py \
# #     ${train_file} \
# #     --work-dir ${work_dir} \
# #     --gpu-ids 1 \
# #### test
python ./MMdet/tools/test.py \
    ${train_file} \
    ${work_dir}/best_bbox_mAP.pth \
    --eval mAP
    # --eval bbox \
    # --options "iou_thrs=[0.5]"
done  

#############################################
##  augment
#############################################

### Weights path

### CLAHE  /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/clahe/best_bbox_mAP.pth
### contrast /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/contrast/best_bbox_mAP.pth
### jitter /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/jitter/best_bbox_mAP.pth
### CLAHE+zero-dce /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/zero_dce_clahe/best_bbox_mAP.pth
### contrast+zero-dce /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/contrast_zero_dce/20211105_203418.log
### jitter zero-dce /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/jitter_zero_dce/best_bbox_mAP.pth


# time=20211103

# train_folder=/home/dlsuncheng/Projects/FSOD/FsMMdet/Model
# work_folder=/home/dlsuncheng/Work_dir/FsMMdet/${time}

# for i in clahe contrast jitter zero_dce_clahe contrast_zero_dce jitter_zero_dce

# do  
# train_file=${train_folder}"/full_data/augment/frcn_all_"$i".py"
# work_dir=${work_folder}"/FRCN_all/"$i

# #### train
# # python ./MMdet/tools/train.py \
# #     ${train_file} \
# #     --work-dir ${work_dir} \
# #     --gpu-ids 1 \
# #### test
# python ./MMdet/tools/test.py \
#     ${train_file} \
#     ${work_dir}/best_bbox_mAP.pth \
#     --eval bbox \
#     --options "iou_thrs=[0.5]"
# done  
