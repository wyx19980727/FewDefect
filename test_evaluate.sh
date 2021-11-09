### test dataset and post analysis
# python ./MMdet/tools/test.py \

# CUDA_VISIBLE_DEVICES=0,1 bash ./MMdet/tools/dist_test.sh \
#     ./Model/full_data/frcn_all.py \
#     /home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_23.pth \
#     2 \
#     --eval bbox \
    # --format-only \
    # --options "jsonfile_prefix=./Result/FRCN_all_test"
    # --show-dir ./Result/FRCN_all_test/Visualize/

# CUDA_VISIBLE_DEVICES=0,1 bash ./MMdet/tools/dist_test.sh \
#     ./Model/full_data/frcn_all_clahe.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/clahe/best_bbox_mAP.pth \
#     2 \
#     --eval bbox \
#     --options "iou_thrs=[0.5]"

# CUDA_VISIBLE_DEVICES=0,1 bash ./MMdet/tools/dist_test.sh \
#     ./Model/full_data/frcn_all_zero_dce.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211102/FRCN_all/zero_dce/best_bbox_mAP.pth \
#     2 \
#     --eval bbox \
#     --options "iou_thrs=[0.5]"

CUDA_VISIBLE_DEVICES=0,1 bash ./MMdet/tools/dist_test.sh \
    ./Model/full_data/frcn_all_zero_dce_clahe.py \
    /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/zero_dce_clahe/best_bbox_mAP.pth \
    2 \
    --eval bbox \
    # proposal \
    --options "iou_thrs=[0.5]"