import torch
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pth1_path",type=str,default="/home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_cos-ft/epoch_24.pth",
                        help="path for the pth file")
    parser.add_argument('--pth2_path', type=str, default='',help='Path to the secondary checkpoint (for combining)')
    parser.add_argument("--save_path",type=str,default="/home/dlsuncheng/FSOD/FsMMdet/Weights/weights_removed.pth",
                        help="path for the pth file")           
    
    parser.add_argument("param_name",type=str,default='roi_head.bbox_head',help="name of the parameters to be deleted")

    parser.add_argument('--method', choices=['combine', 'remove', 'randinit'],
                        required=True,
                        help='Surgery method. combine = combine checkpoints. '
                             'remove = for fine-tuning on novel dataset, remove the final layer of the base detector. '
                             'randinit = randomly initialize novel weights.')
    args = parser.parse_args()
    return args

def reset_ckpt(ckpt):
    if 'meta' in ckpt:
        del ckpt['meta']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0

def combine_ckpt(param_list, is_weight, CLASSNUM, ckpt1, ckpt2=None):

    # weight_name = param_name + ('.weight' if is_weight else '.bias')
    for weight_name in param_list:
        pretrained_weight = ckpt1['state_dict'][weight_name]
        prev_cls = pretrained_weight.size(0)

        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            tar_size = CLASSNUM + 1
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
        else:
            tar_size = CLASSNUM*4
            new_weight = torch.zeros(tar_size)

        new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        ckpt2_weight = ckpt2['state_dict'][weight_name]
        if 'cls_score' in param_name:
            new_weight[prev_cls:-1] = ckpt2_weight[:-1]
            new_weight[-1] = pretrained_weight[-1]
        else:
            new_weight[prev_cls:] = ckpt2_weight
        ckpt1['model'][weight_name] = new_weight

def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))

if __name__ == "__main__":
    args = parse_args()
    pth1_path = args.pth1_path
    ckpt_1 = torch.load(pth1_path)
    save_path = args.save_path
    reset_ckpt(ckpt_1)
    param_name = args.param_name



    if args.method =="remove":


        for parameter in list(ckpt_1["state_dict"].keys()):
            if "fc_cls" in parameter or "fc_reg" in parameter:
                del ckpt_1['state_dict'][parameter]

        save_ckpt(ckpt_1,save_path)

    elif args.method == "combine":

        CLASSNUM = 6

        parameters_to_combine = []
        for parameter in list(ckpt_1["state_dict"].keys()):
            if param_name+"fc_cls" in parameter or param_name+"fc_reg" in parameter:
                parameters_to_combine.append(parameter)
        
        ckpt_2 = torch.load(args.pth2_path)

        for idx, param_name in enumerate(parameters_to_combine):
            if "cls" in param_name:
                combine_ckpt(param_name, True, CLASSNUM, ckpt_1, ckpt_2)
            elif "reg" in param_name:
                combine_ckpt(param_name, False, CLASSNUM, ckpt_1, ckpt_2)
    
        save_ckpt(ckpt_1,save_path)

    elif args.method == "randinit":
        pass

# roi_head.bbox_head.fc_cls.weight
# torch.Size([7, 1024])
# torch.Size([4, 1024])
# roi_head.bbox_head.fc_cls.bias
# torch.Size([7])
# torch.Size([4])
# roi_head.bbox_head.fc_reg.weight
# torch.Size([24, 1024])
# torch.Size([12, 1024])
# roi_head.bbox_head.fc_reg.bias
# torch.Size([24])
# torch.Size([12])
# roi_head.bbox_head.shared_fcs.0.weight
# torch.Size([1024, 12544])
# torch.Size([1024, 12544])
# roi_head.bbox_head.shared_fcs.0.bias
# torch.Size([1024])
# torch.Size([1024])
# roi_head.bbox_head.shared_fcs.1.weight
# torch.Size([1024, 1024])
# torch.Size([1024, 1024])
# roi_head.bbox_head.shared_fcs.1.bias
# torch.Size([1024])
# torch.Size([1024])