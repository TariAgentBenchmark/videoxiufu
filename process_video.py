
import os
import argparse
import torch
import yaml
import torch_npu
import cv2
import numpy as np
from pathlib import Path
# from IPython.display import display, Image
from basicsr.test import test_pipeline
from basicsr.models import build_model
from basicsr.data import build_dataset, build_dataloader
from tgsr.models.hat_model import HATModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='config file path', required=True, type=str)
    args = parser.parse_args()
    cfg_path = args.config
    print(f'加载配置文件:{cfg_path}')
    opt = yaml.full_load(open(cfg_path, encoding='UTF-8'))
    model = HATModel(opt)
    print('测试环境加载完成')
    
    video_dir = opt['video_dir']
    print(f'待处理视频目录:{video_dir}')
    video_files = sorted(list(Path(video_dir).rglob('*.mp4')))
    print(f'开始运行，待处理视频数量:{len(video_files)}')
    
    for i, video_path in enumerate(video_files):
        video_path = str(video_path)
        print(f'正在处理视频({i}/{len(video_files)}): {video_path}')
        dataset_opts = {
            'name': f'video{i}',
            'type': 'SingleVideoDataset',
            'phase': 'val',
            'video_path': video_path
        }
        data_set = build_dataset(dataset_opts)
        # 检测设备类型并适配数据加载器
        if torch.npu.is_available() and opt['num_gpu'] != 0:
            num_devices = torch.npu.device_count()
        elif torch.cuda.is_available() and opt['num_gpu'] != 0:
            num_devices = torch.cuda.device_count()
        else:
            num_devices = 0
            
        data_loader = build_dataloader(
            data_set, dataset_opts, 
            num_gpu=num_devices, dist=opt['dist'], sampler=None, seed=opt['manual_seed']
        )
        out_path = model.restoration_video(data_loader, opt['output_dir'])
        print(f'视频({i}/{len(video_files)})处理完成，保存为{out_path}')



if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)
    main()
