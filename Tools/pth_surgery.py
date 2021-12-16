import torch

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--pth1_path",type=str,default="/home/dlsuncheng/Work_dir/Steel_Defect/20211021/FRCN_cos-ft/epoch_24.pth",
                        help="path for the pth file")
    parser.add_argument('--pth2_path', type=str, default='',help='Path to the secondary checkpoint (for combining)')
    parser.add_argument("--save_dir",type=str,default="/home/dlsuncheng/FSOD/FsMMdet/Weights/",
                        help="path for the pth file")
    # Surgery method
    parser.add_argument('--method', choices=['combine', 'remove', 'randinit'],
                        required=True,
                        help='Surgery method. combine = '
                             'combine checkpoints. remove = for fine-tuning on '
                             'novel dataset, remove the final layer of the '
                             'base detector. randinit = randomly initialize '
                             'novel weights.')
    # Targets
    parser.add_argument('--param-name', type=str, nargs='+',
                        default=["roi_head.bbox_head.fc_cls",
                                 "roi_head.bbox_head.fc_reg"],
                        help='Target parameter names')

    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')

    args = parser.parse_args()
    return args


def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.

    """

    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        ### 指定参数名称
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        ### 获取对应的参数值
        pretrained_weight = ckpt['state_dict'][weight_name]
        ### 获取输入分类器的矩阵的维度
        prev_cls = pretrained_weight.size(0)
        ### 最后一维度表示背景
        if 'fc_cls' in param_name:
            prev_cls -= 1
        
        ### 对参数重新初始化
        ### weights参数随机初始化
        ### bias参数置零
        ### 矩阵的维度发生了变化，主要是由于 base class 和novel class 的类别数量不同
        ### 将矩阵的维度调整为符合base+novel class的数量
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)
        
        ### 把旧矩阵的权重传递给新矩阵
        ### 这里是在原有权重的基础上进行了扩充，在原有维度的基础上，加入了新类别的维度
        ### 得到的新类别的权重=旧类别的部分(复制)+新增类别的部分随机初始化+背景类别(针对cls的参数)
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        if 'fc_cls' in param_name:
            new_weight[-1] = pretrained_weight[-1]  # bg class
        
        ckpt['state_dict'][weight_name] = new_weight

    surgery_loop(args, surgery)


def combine_ckpts(args):
    """
    Combine base detector with novel detector. 
    Feature extractor weights are from the base detector. Only the final layer weights are combined.
    """
    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):

        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['state_dict'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'fc_cls' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
        else:
            new_weight = torch.zeros(tar_size)

        new_weight[:prev_cls] = pretrained_weight[:prev_cls]

        ckpt2_weight = ckpt2['state_dict'][weight_name]

        if 'fc_cls' in param_name:
            new_weight[prev_cls:-1] = ckpt2_weight[:-1]
            new_weight[-1] = pretrained_weight[-1]
        else:
            new_weight[prev_cls:] = ckpt2_weight
        ckpt['state_dict'][weight_name] = new_weight

    surgery_loop(args, surgery)

def surgery_loop(args, surgery):
    # Load checkpoints
    ckpt = torch.load(args.pth1_path)
    ### 是否需要load 第二个权重
    ### 并设置保存的路径
    if args.method == 'combine':
        ckpt2 = torch.load(args.pth2_path)
        save_name = args.tar_name + '_combine.pth'
    else:
        ckpt2 = None
        save_name = args.tar_name + '_' + \
            ('remove' if args.method == 'remove' else 'surgery') + '.pth'

    if args.save_dir == '':
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir

    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)

    ### 删除权重文件中的具体训练信息，包括优化器、训练轮次等
    reset_ckpt(ckpt)

    # 删除权重
    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['state_dict'][param_name + '.weight']
            if param_name+'.bias' in ckpt['state_dict']:
                del ckpt['state_dict'][param_name+'.bias']
        save_ckpt(ckpt, save_path)
        return

    # Surgery
    ## 输入维度，+1 用于判断是否为背景，
    ## *4是预测框的位置回归
    tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]


    for idx, (param_name, tar_size) in enumerate(zip(args.param_name,
                                                     tar_sizes)):
        surgery(param_name, True, tar_size, ckpt, ckpt2)
        surgery(param_name, False, tar_size, ckpt, ckpt2)

    
    # Save to file
    save_ckpt(ckpt, save_path)

def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))

def reset_ckpt(ckpt):
    if 'meta' in ckpt:
        del ckpt['meta']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0


if __name__ == '__main__':
    args = parse_args()

    TAR_SIZE = 6

    if args.method == 'combine':
        combine_ckpts(args)
    else:
        ckpt_surgery(args)

        ### ckpt 函数执行顺序：
        ### 导入权重 ——> 删除具体训练信息 ——> 删除指定权重
        ###                          |
        ###                           ——> 对指定参数，随机初始化：修改维度，