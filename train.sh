#### train dataset
# CUDA_VISIBLE_DEVICES=0,3 bash ./MMdet/tools/dist_train.sh \
#     ./Model/full_data/frcn_all.py 2

#### remove fc layer on new-generated weights file


#### fine-tune on novel class
python ./MMdet/tools/train.py \
    ./Model/few_shot/fc_cos-ft/frcn_cos_unfreeze_novel_30shot.py --gpu-ids 0

# CUDA_VISIBLE_DEVICES=2,3 bash ./MMdet/tools/dist_train.sh \
#     ./Model/few_shot/cos-ft/frcn_cos_unfreeze_novel_1shot.py 2

# CUDA_VISIBLE_DEVICES=2,3 bash ./MMdet/tools/dist_train.sh \
#     ./Model/few_shot/cos-ft/frcn_cos_unfreeze_novel_1shot.py 2

# CUDA_VISIBLE_DEVICES=2,3 bash ./MMdet/tools/dist_train.sh \
#     ./Model/few_shot/cos-ft/frcn_cos_unfreeze_novel_1shot.py 2


# python ./MMdet/tools/train.py \
#     ./Model/few_shot/cos-ft/frcn_cos_unfreeze_novel_30shot.py \
#     --gpu-ids 3

### test dataset and post analysis
# CUDA_VISIBLE_DEVICES=2,3 bash ./MMdet/tools/dist_test.sh \
#     ./Model/cos-ft/frcn_all.py \
#     /home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_21.pth \
#     2 \
#     --eval bbox \
#     --options "jsonfile_prefix=./Result/FRCN_all_test"
    # --options "classwise=True"
    # --show-dir ./Result/FRCN_all_test/Visualize/