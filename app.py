"""
FastAPI主服务 - 多卡视频超分辨率处理
接收视频文件路径或视频流，使用Celery分布式处理，返回处理后的视频文件路径
"""
import os
import asyncio
import uuid
import base64
from pathlib import Path
from typing import Optional, AsyncGenerator

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator

from video_processor import MultiGPUVideoProcessor


app = FastAPI(title="多卡视频超分辨率服务")

# 全局视频处理器
video_processor: Optional[MultiGPUVideoProcessor] = None


class VideoProcessRequest(BaseModel):
    """视频处理请求"""
    video_url: str  # RTSP流地址、视频URL或本地文件路径
    down_sample: bool = True  # 是否下采样
    segment_duration: float = 30.0  # 分段时长（秒），默认30秒
    segment_max_frames: int = 600  # 分段最大帧数，默认600帧
    
    @validator('video_url')
    def validate_video_url(cls, v):
        """验证视频URL或本地文件路径"""
        # 检查是否为流地址或在线视频URL
        if v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return v
        
        # 检查是否为本地文件路径
        if os.path.isfile(v):
            # 验证文件扩展名是否为视频格式
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
            file_ext = Path(v).suffix.lower()
            if file_ext not in video_extensions:
                raise ValueError(f"不支持的视频格式: {file_ext}。支持的格式: {', '.join(video_extensions)}")
            return v
        
        # 如果不是URL也不是存在的文件，则检查是否为有效的路径格式
        try:
            path = Path(v)
            if path.is_absolute() or any(part in v for part in ['/', '\\']):
                raise ValueError(f"本地文件不存在: {v}")
        except Exception:
            pass
        
        raise ValueError("请提供有效的RTSP流地址、在线视频URL或本地视频文件路径")


class SegmentProcessResult(BaseModel):
    """视频片段处理结果"""
    video_id: str  # 视频ID（对应task_id）
    segment_id: int  # 片段序号
    segment_start_time: float  # 片段开始时间（秒）
    segment_end_time: float  # 片段结束时间（秒）
    segment_frame_count: int  # 片段包含的帧数
    before_first_frame: str  # 处理前第一帧图像（base64编码）
    after_first_frame: str  # 处理后第一帧图像（base64编码）
    before_segment_url: str  # 处理前视频片段保存路径
    after_segment_url: str  # 处理后视频片段保存路径
    processing_time: float  # 处理耗时（秒）


@app.on_event("startup")
async def startup_event():
    """启动时初始化视频处理器"""
    global video_processor
    
    # 创建临时存储目录
    os.makedirs("data/temp_frames", exist_ok=True)
    
    # 初始化多GPU视频处理器
    video_processor = MultiGPUVideoProcessor(
        config_path="options/video.yml",
        num_gpus=8,  # 8张计算卡
        gpu_devices=list(range(8))  # GPU设备ID: 0-7
    )
    
    print("多卡视频流式处理服务启动完成")


@app.post("/process_video")
async def process_video(request: VideoProcessRequest):
    """
    分段处理视频接口 - 使用SSE返回每个视频片段的处理结果
    
    Args:
        request: 包含视频URL、下采样选项、分段配置等的请求对象
        
    Returns:
        SSE流，每个片段处理完成后推送JSON结果
    """
    if video_processor is None:
        raise HTTPException(status_code=500, detail="视频处理器未初始化")
    
    # 生成视频ID
    video_id = str(uuid.uuid4())
    
    async def generate_stream():
        """生成SSE流"""
        async for segment_result in video_processor.process_video_segments(
            video_url=request.video_url,
            video_id=video_id,
            down_sample=request.down_sample,
            segment_duration=request.segment_duration,
            segment_max_frames=request.segment_max_frames
        ):
            # 返回SSE格式数据
            yield f"data: {segment_result.json()}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "多卡视频超分辨率服务",
        "processor_ready": video_processor is not None
    }


# 流式处理模式下不再需要任务状态查询


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)