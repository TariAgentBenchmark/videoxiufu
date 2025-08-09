"""
多GPU视频流式处理器 - 实时处理视频帧
"""
import os
import asyncio
import tempfile
import base64
from typing import List, Dict, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from celery import group

from celery_app import celery_app
from worker import process_frame_task

# 定义片段处理结果类（避免循环导入）
class SegmentProcessResult:
    def __init__(self, video_id: str, segment_id: int, segment_start_time: float,
                 segment_end_time: float, segment_frame_count: int, before_first_frame: str,
                 after_first_frame: str, before_segment_url: str, after_segment_url: str,
                 processing_time: float):
        self.video_id = video_id
        self.segment_id = segment_id
        self.segment_start_time = segment_start_time
        self.segment_end_time = segment_end_time
        self.segment_frame_count = segment_frame_count
        self.before_first_frame = before_first_frame
        self.after_first_frame = after_first_frame
        self.before_segment_url = before_segment_url
        self.after_segment_url = after_segment_url
        self.processing_time = processing_time
    
    def json(self):
        """返回JSON格式字符串"""
        import json
        return json.dumps({
            "video_id": self.video_id,
            "segment_id": self.segment_id,
            "segment_start_time": self.segment_start_time,
            "segment_end_time": self.segment_end_time,
            "segment_frame_count": self.segment_frame_count,
            "before_first_frame": self.before_first_frame,
            "after_first_frame": self.after_first_frame,
            "before_segment_url": self.before_segment_url,
            "after_segment_url": self.after_segment_url,
            "processing_time": self.processing_time
        })


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
    
    async def process_video_segments(self, video_url: str, video_id: str, 
                                   down_sample: bool = True, segment_duration: float = 30.0,
                                   segment_max_frames: int = 600) -> AsyncGenerator[SegmentProcessResult, None]:
        """
        分段处理视频，每处理完一个片段就返回结果
        
        Args:
            video_url: 视频URL或路径
            video_id: 视频ID
            down_sample: 是否下采样
            segment_duration: 片段时长（秒）
            segment_max_frames: 片段最大帧数
            
        Yields:
            SegmentProcessResult: 每个片段的处理结果
        """
        import time
        
        # 创建输出目录
        os.makedirs("data/output_segments", exist_ok=True)
        os.makedirs("data/temp_frames", exist_ok=True)
        
        # 获取视频信息
        video_info = await self._get_video_info(video_url)
        fps = video_info.get("fps", 30)
        
        # 计算每个片段的帧数
        frames_per_segment = min(int(segment_duration * fps), segment_max_frames)
        
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_url}")
        
        segment_id = 0
        
        try:
            while True:
                start_time = time.time()
                
                # 提取当前片段的帧
                segment_frames = []
                frame_idx = 0
                
                while frame_idx < frames_per_segment:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # 编码帧为bytes
                    ret_encode, buffer = cv2.imencode('.jpg', frame)
                    if ret_encode:
                        segment_frames.append((frame_idx, buffer.tobytes()))
                        frame_idx += 1
                
                # 如果没有更多帧，退出
                if not segment_frames:
                    break
                
                # 计算片段时间信息
                segment_start_time = segment_id * segment_duration
                segment_end_time = segment_start_time + (len(segment_frames) / fps)
                
                # 分布式处理当前片段的帧
                processed_frames = await self._process_frames_distributed(
                    segment_frames, down_sample, f"{video_id}_segment_{segment_id}"
                )
                
                if not processed_frames:
                    segment_id += 1
                    continue
                
                # 保存原始片段视频
                before_segment_path = f"data/output_segments/{video_id}_segment_{segment_id}_before.mp4"
                await self._save_segment_video(segment_frames, before_segment_path, fps)
                
                # 保存处理后片段视频
                after_segment_path = f"data/output_segments/{video_id}_segment_{segment_id}_after.mp4"
                await self._save_segment_video(processed_frames, after_segment_path, fps)
                
                # 根据环境变量返回绝对路径或相对路径
                mount_path = os.environ.get('DATA_MOUNT_PATH')
                if mount_path:
                    # 如果设置了挂载路径环境变量，返回绝对路径
                    before_segment_url = f"{mount_path}/output_segments/{video_id}_segment_{segment_id}_before.mp4"
                    after_segment_url = f"{mount_path}/output_segments/{video_id}_segment_{segment_id}_after.mp4"
                else:
                    # 否则返回相对路径
                    before_segment_url = before_segment_path
                    after_segment_url = after_segment_path
                
                # 获取第一帧的base64编码（处理前后对比）
                before_first_frame_b64 = self._frame_to_base64(segment_frames[0][1])
                after_first_frame_b64 = self._frame_to_base64(processed_frames[0][1])
                
                # 计算处理时间
                processing_time = time.time() - start_time
                
                # 生成片段处理结果
                segment_result = SegmentProcessResult(
                    video_id=video_id,
                    segment_id=segment_id,
                    segment_start_time=segment_start_time,
                    segment_end_time=segment_end_time,
                    segment_frame_count=len(processed_frames),
                    before_first_frame=before_first_frame_b64,
                    after_first_frame=after_first_frame_b64,
                    before_segment_url=before_segment_url,
                    after_segment_url=after_segment_url,
                    processing_time=processing_time
                )
                
                yield segment_result
                segment_id += 1
                
        finally:
            cap.release()
    
    async def _save_segment_video(self, frames_data: List[tuple], output_path: str, fps: float):
        """
        保存视频片段
        
        Args:
            frames_data: 帧数据列表 [(frame_idx, frame_bytes), ...]
            output_path: 输出路径
            fps: 帧率
        """
        def save_sync():
            if not frames_data:
                return
            
            # 获取第一帧来确定视频尺寸
            first_frame_data = frames_data[0][1]
            nparr = np.frombuffer(first_frame_data, np.uint8)
            first_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            height, width = first_frame.shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise ValueError(f"无法创建视频写入器: {output_path}")
            
            # 写入所有帧
            for frame_idx, frame_data in frames_data:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is not None:
                    video_writer.write(frame)
            
            video_writer.release()
        
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, save_sync)
    
    def _frame_to_base64(self, frame_bytes: bytes) -> str:
        """
        将帧数据转换为base64编码
        
        Args:
            frame_bytes: 帧的bytes数据
            
        Returns:
            base64编码的字符串
        """
        return base64.b64encode(frame_bytes).decode('utf-8')