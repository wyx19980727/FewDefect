#### pre-train on base dataset
# python ./MMdet/tools/train.py \
#     ./Model/base_pretrain/frcn_fc_base_pretrain.py --gpu-ids 0

#### remove fc layer on new-generated weights file

# python /home/dlsuncheng/FSOD/FsMMdet/Tools/pth_surgery.py \
#     --pth1_path /home/dlsuncheng/Work_dir/FsMMdet/20211101/20211101/fc_base_pretrain/best_bbox_mAP_epoch_19.pth \
#     --method "remove"


#### fine-tune on novel class
# python ./MMdet/tools/train.py \
#     ./Model/few_shot/fc_cos-ft/frcn_cos_unfreeze_novel_30shot.py --gpu-ids 0

### combine methods

# python /home/dlsuncheng/FSOD/FsMMdet/Tools/pth_surgery.py \
#     --pth1_path /home/dlsuncheng/Work_dir/FsMMdet/20211101/20211101/fc_base_pretrain/best_bbox_mAP_epoch_19.pth \
#     --pth2_path /home/dlsuncheng/Work_dir/FsMMdet/20211101/FRCN_fc_cos-ft/novel_unfreeze_30shot/best_bbox_mAP.pth \
#     --method "combine"

### fine tune on all dataset

# python ./MMdet/tools/train.py \
#     ./Model/full_data/frcn_cos_unfreeze_all_30shot.py \
#     --gpu-ids 0

### test dataset and post analysis
# CUDA_VISIBLE_DEVICES=2,3 bash ./MMdet/tools/dist_test.sh \
# python ./MMdet/tools/test.py \
#     ./Model/full_data/frcn_cos_unfreeze_all_30shot.py \
#     /home/dlsuncheng/Work_dir/FsMMdet/20211101/FRCN_fc_cos-ft/all_unfrozen_30shot/best_bbox_mAP.pth \
#     --eval bbox \
#     --options "jsonfile_prefix=./Result/FRCN_all_test"
    # --options "classwise=True"
    # --show-dir ./Result/FRCN_all_test/Visualize/



## check data augmentation result
# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_clahe.py \
#     --gpu-ids 3

# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_retinex.py \
#     --gpu-ids 3

# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_zero_dce_clahe.py \
#     --gpu-ids 3


# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_zero_dce_retinex.py \
#     --gpu-ids 3

# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_retinex_clahe.py \
#     --gpu-ids 3

# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_clahe_retinex.py \
#     --gpu-ids 3

# python ./MMdet/tools/train.py \
#     /home/dlsuncheng/FSOD/FsMMdet/Model/full_data/frcn_all_jitter_zero_dce.py \
#     --gpu-ids 1

CUDA_VISIBLE_DEVICES=0,1 bash ./MMdet/tools/dist_test.sh \
    ./Model/full_data/frcn_all_jitter_zero_dce.py \
    /home/dlsuncheng/Work_dir/FsMMdet/20211103/FRCN_all/jitter_zero_dce/best_bbox_mAP.pth \
    2 \
    --eval bbox \
    --options "iou_thrs=[0.5]"
