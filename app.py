"""
FastAPI主服务 - 多卡视频超分辨率处理
接收视频文件路径或视频流，使用Celery分布式处理，返回处理后的视频文件路径
"""
import os
import asyncio
import uuid
from pathlib import Path
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from video_processor import MultiGPUVideoProcessor


app = FastAPI(title="多卡视频超分辨率服务")

# 全局视频处理器
video_processor: Optional[MultiGPUVideoProcessor] = None


class VideoProcessRequest(BaseModel):
    """视频处理请求"""
    video_path: str  # 视频文件绝对路径或RTSP流地址
    down_sample: bool = True  # 是否下采样
    output_dir: str = "output"  # 输出目录


class VideoProcessResponse(BaseModel):
    """视频处理响应"""
    task_id: str  # 任务ID
    output_path: str  # 输出视频文件绝对路径
    status: str  # 处理状态


@app.on_event("startup")
async def startup_event():
    """启动时初始化视频处理器"""
    global video_processor
    
    # 检查输出目录
    os.makedirs("output", exist_ok=True)
    
    # 初始化多GPU视频处理器
    video_processor = MultiGPUVideoProcessor(
        config_path="options/video.yml",
        num_gpus=8,  # 8张计算卡
        gpu_devices=list(range(8))  # GPU设备ID: 0-7
    )
    
    print("多卡视频处理服务启动完成")


@app.post("/process_video", response_model=VideoProcessResponse)
async def process_video(request: VideoProcessRequest):
    """
    处理视频接口
    
    Args:
        request: 包含视频路径、下采样选项等的请求对象
        
    Returns:
        包含任务ID和输出路径的响应对象
    """
    if video_processor is None:
        raise HTTPException(status_code=500, detail="视频处理器未初始化")
    
    # 验证输入视频
    if not request.video_path.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        # 本地文件路径
        if not os.path.exists(request.video_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {request.video_path}")
    
    # 生成任务ID和输出路径
    task_id = str(uuid.uuid4())
    output_filename = f"processed_{task_id}.mp4"
    output_path = os.path.abspath(os.path.join(request.output_dir, output_filename))
    
    # 确保输出目录存在
    os.makedirs(request.output_dir, exist_ok=True)
    
    try:
        # 执行视频处理
        result_path = await video_processor.process_video(
            video_path=request.video_path,
            output_path=output_path,
            down_sample=request.down_sample,
            task_id=task_id
        )
        
        return VideoProcessResponse(
            task_id=task_id,
            output_path=result_path,
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"视频处理失败: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "多卡视频超分辨率服务",
        "processor_ready": video_processor is not None
    }


@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    if video_processor is None:
        raise HTTPException(status_code=500, detail="视频处理器未初始化")
    
    status = video_processor.get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return {"task_id": task_id, "status": status}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)