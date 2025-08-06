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
    video_url: str  # RTSP流地址或视频URL
    down_sample: bool = True  # 是否下采样
    
    @validator('video_url')
    def validate_video_url(cls, v):
        """验证视频URL必须是流地址或在线视频"""
        if not v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise ValueError("只支持RTSP流和在线视频URL，不支持本地文件路径")
        return v


class FrameProcessResult(BaseModel):
    """单帧处理结果"""
    video_id: str  # 视频ID（对应task_id）
    segment_id: int  # 帧序号
    before_frame: str  # 处理前的帧图像（base64编码）
    after_frame: str  # 处理后的帧图像（base64编码）
    before_segment_url: str  # 处理前帧的临时存储路径
    after_segment_url: str  # 处理后帧的临时存储路径


@app.on_event("startup")
async def startup_event():
    """启动时初始化视频处理器"""
    global video_processor
    
    # 创建临时存储目录
    os.makedirs("temp_frames", exist_ok=True)
    
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
    流式处理视频接口 - 使用SSE返回每帧处理结果
    
    Args:
        request: 包含视频URL、下采样选项等的请求对象
        
    Returns:
        SSE流，每帧处理完成后推送JSON结果
    """
    if video_processor is None:
        raise HTTPException(status_code=500, detail="视频处理器未初始化")
    
    # 生成视频ID
    video_id = str(uuid.uuid4())
    
    async def generate_stream():
        """生成SSE流"""
        async for frame_result in video_processor.process_video_stream(
            video_url=request.video_url,
            video_id=video_id,
            down_sample=request.down_sample
        ):
            # 返回SSE格式数据
            yield f"data: {frame_result.json()}\n\n"
    
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