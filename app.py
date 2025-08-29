"""
FastAPI主服务 - 多卡视频超分辨率处理
接收视频文件路径或视频流，使用Celery分布式处理，返回处理后的视频文件路径
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, validator

from utils import get_new_url, send_fix_ret, send_fix_stopped
from video_processor import MultiGPUVideoProcessor

app = FastAPI(title="多卡视频超分辨率服务")

# 全局视频处理器
video_processor: Optional[MultiGPUVideoProcessor] = None

# 任务管理
running_tasks: Dict[str, Dict] = {}  # key: task_id(video_id_cor_id), value: task_info
MAX_CONCURRENT_TASKS = 4  # 最大并发任务数


class VideoProcessRequest(BaseModel):
    """视频处理请求"""

    video_url: str  # RTSP流地址、视频URL或本地文件路径
    down_sample: bool = True  # 是否下采样
    segment_duration: float = 10.0  # 分段时长（秒），默认30秒
    segment_max_frames: int = 150  # 分段最大帧数，默认600帧

    @validator("video_url")
    def validate_video_url(cls, v):
        """验证视频URL或本地文件路径"""
        # 检查是否为流地址或在线视频URL
        if v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            return v

        # 检查是否为本地文件路径
        if os.path.isfile(v):
            # 验证文件扩展名是否为视频格式
            video_extensions = {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
                ".flv",
                ".wmv",
                ".m4v",
                ".webm",
            }
            file_ext = Path(v).suffix.lower()
            if file_ext not in video_extensions:
                raise ValueError(
                    f"不支持的视频格式: {file_ext}。支持的格式: {', '.join(video_extensions)}"
                )
            return v

        # 如果不是URL也不是存在的文件，则检查是否为有效的路径格式
        try:
            path = Path(v)
            if path.is_absolute() or any(part in v for part in ["/", "\\"]):
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
        gpu_devices=list(range(8)),  # GPU设备ID: 0-7
    )

    print("多卡视频流式处理服务启动完成")


async def process_video_task(task_id: str, task_info: Dict):
    """
    后台视频处理任务

    Args:
        task_id: 任务ID
        task_info: 任务信息字典
    """
    video_id = task_info["video_id"]
    cor_id = task_info["cor_id"]
    video_url = task_info["video_url"]
    token = task_info["token"]

    try:
        print(f"开始处理视频任务: {task_id}")

        # 使用视频处理器分段处理视频
        async for segment_result in video_processor.process_video_segments(
            video_url=video_url,
            video_id=video_id,
            down_sample=True,  # 默认下采样
            segment_duration=10.0,  # 10秒分段
            segment_max_frames=150,  # 最大150帧
        ):
            # 检查任务是否被取消
            if task_id in running_tasks and running_tasks[task_id]["cancel_flag"]:
                print(f"任务 {task_id} 被取消")
                break

            # 构造推送给前端的结果
            result_data = {
                "video_id": video_id,
                "cor_id": cor_id,
                "segment_id": segment_result.segment_id,
                "before_segment_url": segment_result.before_segment_url,
                "after_segment_url": segment_result.after_segment_url,
                "before_frame": segment_result.before_first_frame,  # base64编码的图像
                "after_frame": segment_result.after_first_frame,  # base64编码的图像
            }

            # 推送结果到前端
            if token:
                try:
                    await send_fix_ret(result_data, token)
                    print(f"成功推送片段 {segment_result.segment_id} 结果到前端")
                except Exception as e:
                    print(f"推送片段结果失败: {e}")
            else:
                print(f"无token，跳过推送片段 {segment_result.segment_id}")

        # 任务完成后的清理
        if task_id in running_tasks:
            if not running_tasks[task_id]["cancel_flag"]:
                running_tasks[task_id]["status"] = "completed"
                print(f"视频处理任务 {task_id} 完成")

            # 发送任务完成通知
            if token:
                try:
                    stop_result = {
                        "video_id": video_id,
                        "cor_id": cor_id,
                        "video_url": task_info["original_url"],
                        "description": "视频已结束"
                        if not running_tasks[task_id]["cancel_flag"]
                        else "任务已取消",
                    }
                    await send_fix_stopped(stop_result, token)
                except Exception as e:
                    print(f"发送完成通知失败: {e}")

            # 清理任务记录
            del running_tasks[task_id]

    except Exception as e:
        print(f"视频处理任务 {task_id} 失败: {e}")

        # 错误处理和清理
        if task_id in running_tasks:
            running_tasks[task_id]["status"] = "failed"

            # 发送错误通知
            if token:
                try:
                    stop_result = {
                        "video_id": video_id,
                        "cor_id": cor_id,
                        "video_url": task_info["original_url"],
                        "description": f"处理失败: {str(e)}",
                    }
                    await send_fix_stopped(stop_result, token)
                except Exception as notify_e:
                    print(f"发送错误通知失败: {notify_e}")

            # 清理任务记录
            del running_tasks[task_id]


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
            segment_max_frames=request.segment_max_frames,
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
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "多卡视频超分辨率服务",
        "processor_ready": video_processor is not None,
        "running_tasks": len(running_tasks),
        "max_concurrent_tasks": MAX_CONCURRENT_TASKS,
    }


@app.get("/api/quality/start_fix")
async def start_fix(video_id: str, cor_id: str, video_url: str):
    """
    开启视频质量修复处理

    Args:
        video_id: 视频ID
        cor_id: 关联ID
        video_url: 视频地址（支持RTSP流、HTTP流、本地文件等）

    Returns:
        处理状态响应
    """
    if video_processor is None:
        raise HTTPException(status_code=500, detail="视频处理器未初始化")

    # 生成任务ID
    task_id = f"{video_id}_{cor_id}"

    # 检查任务是否已存在
    if task_id in running_tasks:
        return JSONResponse(content={"description": "正在处理"}, status_code=202)

    # 检查并发任务数量限制
    if len(running_tasks) >= MAX_CONCURRENT_TASKS:
        return JSONResponse(
            content={"description": "已到达并行处理上限，请等待其他任务完成后再请求"},
            status_code=500,
        )

    try:
        # 处理视频URL（如果是需要token的流）
        processed_video_url = video_url
        token = None

        # 检查是否需要token验证的流
        if video_url.startswith(("rtsp://", "rtmp://")) and "tk=" not in video_url:
            try:
                processed_video_url, token = await get_new_url(video_url)
            except Exception as e:
                print(f"获取token失败: {e}")
                # 如果获取token失败，尝试直接使用原URL
                processed_video_url = video_url

        # 记录任务信息
        task_info = {
            "video_id": video_id,
            "cor_id": cor_id,
            "video_url": processed_video_url,
            "original_url": video_url,
            "token": token,
            "status": "starting",
            "task": None,
            "cancel_flag": False,
        }
        running_tasks[task_id] = task_info

        # 启动后台处理任务
        task = asyncio.create_task(process_video_task(task_id, task_info))
        running_tasks[task_id]["task"] = task
        running_tasks[task_id]["status"] = "processing"

        return JSONResponse(content={"description": "请求成功"}, status_code=200)

    except Exception as e:
        print(f"启动视频处理任务失败: {e}")
        # 清理失败的任务记录
        if task_id in running_tasks:
            del running_tasks[task_id]
        return JSONResponse(
            content={"description": f"启动处理失败: {str(e)}"}, status_code=500
        )


# 终止任务的API端点
@app.get("/api/quality/stop_fix")
async def stop_fix(video_id: str, cor_id: str, video_url: str):
    """
    停止视频质量修复处理

    Args:
        video_id: 视频ID
        cor_id: 关联ID
        video_url: 视频地址

    Returns:
        停止状态响应
    """
    # 生成任务ID
    task_id = f"{video_id}_{cor_id}"

    # 检查任务是否存在
    if task_id not in running_tasks:
        return JSONResponse(content={"description": "未发现任务"}, status_code=202)

    try:
        task_info = running_tasks[task_id]

        # 设置取消标志
        running_tasks[task_id]["cancel_flag"] = True
        running_tasks[task_id]["status"] = "cancelling"

        # 取消异步任务（如果存在）
        task = task_info.get("task")
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # 发送停止通知给前端
        token = task_info.get("token")
        if token:
            try:
                stop_result = {
                    "video_id": video_id,
                    "cor_id": cor_id,
                    "video_url": video_url,
                    "description": "任务已停止",
                }
                await send_fix_stopped(stop_result, token)
            except Exception as e:
                print(f"发送停止通知失败: {e}")

        # 清理任务记录
        if task_id in running_tasks:
            del running_tasks[task_id]

        print(f"任务 {task_id} 已停止")
        return JSONResponse(content={"description": "请求成功"}, status_code=200)

    except Exception as e:
        print(f"停止任务 {task_id} 失败: {e}")
        return JSONResponse(
            content={"description": f"停止任务失败: {str(e)}"}, status_code=500
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)