"""
配置管理 - 多GPU环境设置和系统配置
"""
import os
from typing import List, Dict


class MultiGPUConfig:
    """多GPU配置类"""
    
    def __init__(self):
        self.num_gpus = 8  # GPU数量
        self.gpu_devices = list(range(8))  # GPU设备ID列表
        self.config_path = "options/video.yml"  # 模型配置文件路径
        self.pretrained_weights = "pretrained_weights/RealHATGAN-TG.pth"  # 模型权重路径
        
        # 输出目录配置
        self.output_dir = "output"
        self.temp_dir = "temp"
        
        # Celery配置
        self.celery_config = {
            "broker_url": "redis://localhost:6379/0",
            "result_backend": "redis://localhost:6379/1",
            "worker_concurrency": 1,  # 每个GPU一个进程
            "task_time_limit": 300,  # 5分钟任务超时
        }
        
        # 视频处理配置
        self.video_config = {
            "max_frames": 10000,  # 最大处理帧数
            "default_fps": 30,  # 默认FPS
            "video_timeout": 600,  # 视频处理超时（秒）
            "tile_size": 512,  # tile模式的patch大小
        }
    
    def setup_directories(self):
        """创建必要的目录"""
        directories = [self.output_dir, self.temp_dir, "output_frames"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_gpu_env_vars(self, device_id: int) -> Dict[str, str]:
        """
        获取指定GPU的环境变量配置
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            环境变量字典
        """
        return {
            "ASCEND_RT_VISIBLE_DEVICES": str(device_id),
            "GPU_DEVICE_ID": str(device_id),
            "CUDA_VISIBLE_DEVICES": str(device_id),  # 兼容CUDA
        }
    
    def validate_config(self) -> bool:
        """验证配置的有效性"""
        # 检查模型配置文件
        if not os.path.exists(self.config_path):
            print(f"错误: 模型配置文件不存在: {self.config_path}")
            return False
        
        # 检查预训练权重
        if not os.path.exists(self.pretrained_weights):
            print(f"错误: 预训练权重文件不存在: {self.pretrained_weights}")
            return False
        
        # 检查GPU设备列表
        if len(self.gpu_devices) != self.num_gpus:
            print(f"错误: GPU设备数量不匹配: 配置{self.num_gpus}个，实际{len(self.gpu_devices)}个")
            return False
        
        return True


# 全局配置实例
config = MultiGPUConfig()