import torch
import torch.backends.cudnn as cudnn

import os, sys
import cv2
from PIL import Image
import cmapy
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import post_process_depth, flip_lr, compute_errors
from networks.sm4depth_swin import SM4Depth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='SM4Depth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='sm4depth')
parser.add_argument('--kbins', type=int, help='model type', default='4')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='base07')
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset', type=str, help='dataset to train on, kitti, nyu, multi', default='multi')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)

# Preprocessing
parser.add_argument('--do_random_rotate', help='if set, will perform random rotation for augmentation',
                    action='store_true')
parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)

# Eval
parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text for evaluation', required=False)
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'multi':
    from dataloaders.dataloader import SM4DataLoader


def test(args, model, dataloader_eval, post_process=False):
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            save_depth_path = eval_sample_batched['save_depth_path'][0]

            pred_depth, _, w = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped, _, w = model(image_flipped)
                pred_depth = post_process_depth(pred_depth[-1], pred_depth_flipped[-1])

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            pred_depth = cv2.resize(pred_depth, dsize=(gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

            if os.path.exists(save_depth_path):
                os.makedirs(os.path.join(save_depth_path, 'depth'), exist_ok=True)
                os.makedirs(os.path.join(save_depth_path, 'colormap'), exist_ok=True)
                # save depth
                pred_depth = (pred_depth * 256).astype(np.int32)
                depth = Image.fromarray(pred_depth)
                depth.save(os.path.join(save_depth_path, 'depth', 'depth.png'))
                # save depth (add color)
                norm_depth = ((pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min()) * 255).astype(np.uint8)
                norm_depth = cv2.applyColorMap(norm_depth, cmapy.cmap('magma_r'))
                cv2.imwrite(os.path.join(save_depth_path, 'colormap', 'norm_depth.png'), norm_depth)
                
            else:
                print(f"{save_depth_path} is not exists.")
                exit

    print('Done.')
    return None


def main_worker(args):
    model = SM4Depth(version=args.encoder, pretrained=None, kbins=args.kbins)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))

    cudnn.benchmark = True
    dataloader_test = SM4DataLoader(args, 'test')
    model.eval()
    with torch.no_grad():
        eval_measures = test(args, model, dataloader_test, post_process=True)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1
    
    main_worker(args)
    print("Begin")


if __name__ == '__main__':
    main()
