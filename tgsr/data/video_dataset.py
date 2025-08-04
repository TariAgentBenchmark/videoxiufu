import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from basicsr.utils.registry import DATASET_REGISTRY



@DATASET_REGISTRY.register()
class SingleVideoDataset(Dataset):
    def __init__(self, opt):
        assert os.path.isfile(opt['video_path'])
        self.video_path = opt['video_path']
        self.sample_interval = opt.get('sample_interval', 1)
        self.max_frames_num = opt.get('max_frames_num', -1)

        self.video_name = os.path.splitext(os.path.basename(opt['video_path']))[0]
        self.frames = list()
        frame_id = 0
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Error: Could not open video.")
        # 获取视频的帧宽度
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 获取视频的帧高度
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % self.sample_interval == 0:
                self.frames.append(frame)
            if self.max_frames_num != -1 and frame_id >= self.max_frames_num:
                break
            frame_id += 1
        cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = self.frames[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return {
            'lq': img,
            'lq_path': os.path.join(self.video_name + f'/{idx}.jpg'),
        }

