"""
Celery Worker - 多GPU视频帧处理任务
每个GPU一个worker进程，通过环境变量指定GPU设备
"""
import os
import cv2
import numpy as np
import torch
import torch_npu
import yaml
from PIL import Image
from torchvision.transforms.functional import to_tensor

from celery_app import celery_app
from tgsr.models.hat_model import HATModel


class GPUWorker:
    """GPU Worker类 - 封装模型加载和推理逻辑"""
    
    def __init__(self, device_id: int, config_path: str = "options/video.yml"):
        """
        初始化GPU Worker
        
        Args:
            device_id: GPU设备ID
            config_path: 模型配置文件路径
        """
        self.device_id = device_id
        self.config_path = config_path
        self.model = None
        self._setup_device()
        self._load_model()
    
    def _setup_device(self):
        """设置GPU设备"""
        # 设置华为昇腾NPU设备
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device_id)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        print(f"Worker初始化 - GPU设备: {self.device_id}")
    
    def _load_model(self):
        """加载HAT模型"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)
        
        self.model = HATModel(opt)
        self.model.eval()
        print(f"GPU {self.device_id} 模型加载完成")
    
    def process_frame(self, frame_data: bytes, frame_idx: int, down_sample: bool = True) -> bytes:
        """
        处理单帧图像
        
        Args:
            frame_data: 帧图像数据（bytes格式）
            frame_idx: 帧索引
            down_sample: 是否下采样
            
        Returns:
            处理后的帧图像数据（bytes格式）
        """
        try:
            # 将bytes转换为numpy数组
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise ValueError("无法解码图像数据")
            
            # 颜色空间转换
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 下采样（如果需要）
            if down_sample:
                h, w = img.shape[:2]
                img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            
            # 转换为tensor并推理
            img_pil = Image.fromarray(img)
            tensor = to_tensor(img_pil).unsqueeze(0).to(self.model.device)
            
            # 模型推理
            self.model.feed_data({'lq': tensor})
            self.model.pre_process()
            
            if 'tile' in self.model.opt:
                self.model.tile_process()
            else:
                self.model.process()
            
            self.model.post_process()
            
            # 获取结果
            result = self.model.get_current_visuals()['result']
            result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # 编码为bytes
            ret, buffer = cv2.imencode('.jpg', result_bgr)
            if not ret:
                raise ValueError("图像编码失败")
            
            print(f"GPU {self.device_id} 处理完成帧 {frame_idx}")
            return buffer.tobytes()
            
        except Exception as e:
            print(f"GPU {self.device_id} 处理帧 {frame_idx} 失败: {str(e)}")
            raise


# 全局Worker实例（每个进程一个）
_worker_instance = None


def get_worker_instance():
    """获取Worker实例（懒加载）"""
    global _worker_instance
    if _worker_instance is None:
        # 从环境变量获取GPU设备ID
        device_id = int(os.environ.get("GPU_DEVICE_ID", "0"))
        _worker_instance = GPUWorker(device_id)
    return _worker_instance


@celery_app.task(bind=True)
def process_frame_task(self, frame_data: bytes, frame_idx: int, down_sample: bool = True):
    """
    Celery任务 - 处理单帧图像
    
    Args:
        frame_data: 帧图像数据（bytes格式）
        frame_idx: 帧索引
        down_sample: 是否下采样
        
    Returns:
        dict: 包含处理结果的字典
    """
    try:
        worker = get_worker_instance()
        result_data = worker.process_frame(frame_data, frame_idx, down_sample)
        
        return {
            "success": True,
            "frame_idx": frame_idx,
            "result_data": result_data,
            "device_id": worker.device_id
        }
        
    except Exception as e:
        return {
            "success": False,
            "frame_idx": frame_idx,
            "error": str(e),
            "device_id": os.environ.get("GPU_DEVICE_ID", "unknown")
        }