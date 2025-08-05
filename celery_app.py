"""
Celery应用配置 - 支持多GPU分布式视频帧处理
"""
from celery import Celery

# 创建Celery应用实例
celery_app = Celery("video_sr_processor")

# Celery配置
celery_app.conf.update(
    # 使用Redis作为消息代理和结果后端
    broker_url="redis://localhost:6379/0",
    result_backend="redis://localhost:6379/1",
    
    # 任务序列化设置
    task_serializer="pickle",
    accept_content=["pickle", "json"],
    result_serializer="pickle",
    
    # 任务路由配置
    task_routes={
        "worker.process_frame_task": {"queue": "gpu_queue"}
    },
    
    # Worker配置
    worker_prefetch_multiplier=1,  # 每个worker同时处理的任务数
    task_acks_late=True,  # 任务完成后再确认
    worker_disable_rate_limits=True,
    
    # 结果过期时间
    result_expires=3600,  # 1小时
    
    # 任务时间限制
    task_time_limit=300,  # 5分钟
    task_soft_time_limit=240,  # 4分钟软限制
    
    # 并发设置
    worker_concurrency=1,  # 每个worker进程数，由于GPU内存限制设为1
    
    # 多进程启动方法设置 - 华为昇腾NPU需要spawn方法
    worker_pool='solo',  # 使用solo pool避免multiprocessing问题
)