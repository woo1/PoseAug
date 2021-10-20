from __future__ import print_function, absolute_import, division

import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from function_baseline.config import get_parse_args
from function_baseline.model_pos_preparation import model_pos_preparation, model_pos_preparation2
from common.viz import wrap_show3d_pose, wrap_show2d_pose

def main(args):
    print('==> Using settings {}'.format(args))
    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    # print('==> Loading dataset...')
    # data_dict = data_preparation(args)

    print("==> Creating model...")
    model_pos = model_pos_preparation2(args, device)

    # Check if evaluate checkpoint file exist:
    assert path.isfile(args.evaluate), '==> No checkpoint found at {}'.format(args.evaluate)
    print("==> Loading checkpoint '{}'".format(args.evaluate))
    ckpt = torch.load(args.evaluate)
    try:
        model_pos.load_state_dict(ckpt['state_dict'])
    except:
        model_pos.load_state_dict(ckpt['model_pos'])

    model_pos.eval()

    np_path = args.np_path
    input_data = np.load(np_path)['pose2d']
    # print('input_data', input_data)
    print('input_data', input_data.shape)

    print('==> Evaluating...')

    wrap_show2d_pose(input_data, 'result/2d_'+path.splitext(path.basename(np_path))[0]+'.png')

    inputs_2d = torch.tensor(input_data)
    inputs_2d = inputs_2d.to(device).float()
    num_poses = inputs_2d.size(0)
    print('num_poses', num_poses)
    # outputs_3d = model_pos(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
    outputs_3d = model_pos(inputs_2d.view(num_poses, -1)).view(num_poses, -1, 3).cpu()
    outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]  # the output is relative to the 0 joint
    print('outputs_3d', outputs_3d)
    print('outputs_3d', outputs_3d.shape)

    # render_animation
    wrap_show3d_pose(outputs_3d.detach().numpy(), 'result/3d_'+path.splitext(path.basename(np_path))[0]+'.png')

if __name__ == '__main__':
    args = get_parse_args()
    # fix random
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True

    main(args)
