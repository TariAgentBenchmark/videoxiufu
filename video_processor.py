"""
多GPU视频处理器 - 协调视频帧分发和结果收集
"""
import os
import asyncio
import tempfile
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from celery import group

from celery_app import celery_app
from worker import process_frame_task


class MultiGPUVideoProcessor:
    """多GPU视频处理器"""
    
    def __init__(self, config_path: str, num_gpus: int = 8, gpu_devices: List[int] = None):
        """
        初始化多GPU视频处理器
        
        Args:
            config_path: 模型配置文件路径
            num_gpus: GPU数量
            gpu_devices: GPU设备ID列表
        """
        self.config_path = config_path
        self.num_gpus = num_gpus
        self.gpu_devices = gpu_devices or list(range(num_gpus))
        self.task_status: Dict[str, str] = {}  # 任务状态跟踪
        
        print(f"初始化多GPU视频处理器: {num_gpus}个GPU, 设备ID: {self.gpu_devices}")
    
    async def process_video(self, video_path: str, output_path: str, 
                          down_sample: bool = True, task_id: str = None) -> str:
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径或流地址
            output_path: 输出视频路径
            down_sample: 是否下采样
            task_id: 任务ID
            
        Returns:
            输出视频文件路径
        """
        if task_id:
            self.task_status[task_id] = "processing"
        
        try:
            # 第一步：读取视频帧
            frames_data = await self._extract_frames(video_path)
            video_info = await self._get_video_info(video_path)
            
            if not frames_data:
                raise ValueError("无法从视频中提取帧")
            
            print(f"提取到 {len(frames_data)} 帧，开始分布式处理...")
            
            # 第二步：分布式处理帧
            processed_frames = await self._process_frames_distributed(
                frames_data, down_sample, task_id
            )
            
            # 第三步：重组视频
            result_path = await self._assemble_video(
                processed_frames, output_path, video_info
            )
            
            if task_id:
                self.task_status[task_id] = "completed"
            
            print(f"视频处理完成: {result_path}")
            return result_path
            
        except Exception as e:
            if task_id:
                self.task_status[task_id] = "failed"
            raise e
    
    async def _extract_frames(self, video_path: str) -> List[tuple]:
        """
        提取视频帧
        
        Args:
            video_path: 视频路径
            
        Returns:
            帧数据列表，每个元素为(frame_idx, frame_bytes)
        """
        def extract_frames_sync():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            
            frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 将帧编码为bytes
                ret_encode, buffer = cv2.imencode('.jpg', frame)
                if ret_encode:
                    frames.append((frame_idx, buffer.tobytes()))
                    frame_idx += 1
                
                # 限制最大帧数（避免内存不足）
                if frame_idx > 10000:  # 最多处理10000帧
                    print("警告: 视频帧数过多，只处理前10000帧")
                    break
            
            cap.release()
            return frames
        
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            frames = await loop.run_in_executor(executor, extract_frames_sync)
        
        return frames
    
    async def _get_video_info(self, video_path: str) -> Dict:
        """获取视频信息"""
        def get_info_sync():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"fps": 30, "width": 1920, "height": 1080}
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            return {"fps": fps, "width": width, "height": height}
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            info = await loop.run_in_executor(executor, get_info_sync)
        
        return info
    
    async def _process_frames_distributed(self, frames_data: List[tuple], 
                                        down_sample: bool, task_id: str = None) -> List[tuple]:
        """
        分布式处理帧
        
        Args:
            frames_data: 帧数据列表
            down_sample: 是否下采样
            task_id: 任务ID
            
        Returns:
            处理后的帧数据列表，排序后的(frame_idx, result_bytes)
        """
        if not frames_data:
            return []
        
        # 创建Celery任务组
        job = group(
            process_frame_task.s(frame_bytes, frame_idx, down_sample) 
            for frame_idx, frame_bytes in frames_data
        )
        
        # 提交任务并等待结果
        print(f"提交 {len(frames_data)} 个帧处理任务到 {self.num_gpus} 个GPU...")
        result = job.apply_async()
        
        # 等待所有任务完成
        results = result.get(timeout=600)  # 10分钟超时
        
        # 处理结果
        processed_frames = []
        failed_frames = []
        
        for res in results:
            if res["success"]:
                processed_frames.append((res["frame_idx"], res["result_data"]))
            else:
                failed_frames.append(res["frame_idx"])
                print(f"帧 {res['frame_idx']} 处理失败: {res['error']}")
        
        if failed_frames:
            print(f"警告: {len(failed_frames)} 帧处理失败")
        
        # 按帧索引排序
        processed_frames.sort(key=lambda x: x[0])
        
        print(f"成功处理 {len(processed_frames)} 帧")
        return processed_frames
    
    async def _assemble_video(self, processed_frames: List[tuple], 
                            output_path: str, video_info: Dict) -> str:
        """
        重组视频
        
        Args:
            processed_frames: 处理后的帧数据
            output_path: 输出路径
            video_info: 视频信息
            
        Returns:
            输出视频文件路径
        """
        def assemble_sync():
            if not processed_frames:
                raise ValueError("没有处理后的帧数据")
            
            # 获取第一帧来确定视频尺寸
            first_frame_data = processed_frames[0][1]
            nparr = np.frombuffer(first_frame_data, np.uint8)
            first_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = video_info.get("fps", 30)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise ValueError(f"无法创建视频写入器: {output_path}")
            
            # 写入所有帧
            for frame_idx, frame_data in processed_frames:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
            return output_path
        
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result_path = await loop.run_in_executor(executor, assemble_sync)
        
        return result_path
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """获取任务状态"""
        return self.task_status.get(task_id)
    
    def cleanup_task(self, task_id: str):
        """清理任务状态"""
        if task_id in self.task_status:
            del self.task_status[task_id]