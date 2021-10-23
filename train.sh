#### train dataset
# CUDA_VISIBLE_DEVICES=0,3 bash ./MMdet/tools/dist_train.sh \
#     ./Model/cos-ft/frcn_pretrain.py 2

# CUDA_VISIBLE_DEVICES=0,3 bash ./MMdet/tools/dist_train.sh \
#     ./Model/cos-ft/frcn_all.py 2


### test dataset and post analysis
# CUDA_VISIBLE_DEVICES=2,3 bash ./MMdet/tools/dist_test.sh \
#     ./Model/cos-ft/frcn_all.py \
#     /home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_all/epoch_21.pth \
#     2 \
#     --eval bbox \
#     --options "jsonfile_prefix=./Result/FRCN_all_test"
    # --options "classwise=True"
    # --show-dir ./Result/FRCN_all_test/Visualize/