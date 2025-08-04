# 多卡视频超分辨率系统改动说明

## 概述
将原有的单卡视频处理系统改造为支持8卡并行的分布式处理系统，使用FastAPI + Celery架构，显著提升处理效率。

## 系统架构

```
[FastAPI主服务] -> [Celery任务队列] -> [8个GPU Workers]
       |                                      |
   接收视频流                             并行处理帧
       |                                      |
   [视频帧分发] <- [结果收集重组] <- [返回处理结果]
```

## 新增文件

### 1. `app.py` - FastAPI主服务
**功能**: 
- 接收视频文件路径或RTSP流
- 协调多GPU处理流程
- 返回处理后的视频文件路径

**主要接口**:
- `POST /process_video` - 视频处理接口
- `GET /health` - 健康检查
- `GET /status/{task_id}` - 任务状态查询

### 2. `celery_app.py` - Celery应用配置
**功能**:
- 配置Celery分布式任务队列
- 使用内存作为消息代理（memory backend）
- 任务路由和超时设置

### 3. `worker.py` - GPU Worker实现
**功能**:
- 封装单GPU模型加载和推理逻辑
- 通过环境变量`ASCEND_RT_VISIBLE_DEVICES`指定GPU设备
- 处理单帧图像的Celery任务

**关键特性**:
- 每个Worker绑定一个GPU设备
- 懒加载模型实例
- 支持华为昇腾NPU

### 4. `video_processor.py` - 视频处理器
**功能**:
- 视频帧提取和分发
- 协调多GPU并行处理
- 结果收集和视频重组

**处理流程**:
1. 提取视频帧 -> 2. 分布式处理 -> 3. 重组视频

### 5. `config.py` - 配置管理
**功能**:
- 多GPU环境变量配置
- 系统参数设置
- 配置验证

### 6. `start_workers.py` - Worker启动脚本
**功能**:
- 为每个GPU启动独立的Worker进程
- 设置GPU设备环境变量
- 进程管理和监控

### 7. `test_multi_gpu.py` - 测试脚本
**功能**:
- 测试多GPU处理功能
- 性能基准测试
- 系统健康检查

## 主要改动点

### 1. 模型加载方式改动
**原来**: 在主进程中直接加载模型
```python
sr_model = HATModel(opt)
```

**现在**: 每个GPU Worker独立加载模型
```python
# 在worker.py中
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
self.model = HATModel(opt)
```

### 2. 视频处理流程改动
**原来**: 逐帧同步处理
```python
for frame in video:
    result = process_frame(frame)
    save_frame(result)
```

**现在**: 并行分布式处理
```python
# 提取所有帧
frames = extract_frames(video)
# 分发到多GPU处理
job = group(process_frame_task.s(frame) for frame in frames)
results = job.apply_async().get()
# 重组视频
assemble_video(results)
```

### 3. GPU设备管理改动
**原来**: 自动使用默认GPU
```python
tensor.to(model.device)
```

**现在**: 显式指定GPU设备
```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
```

## 使用方法

### 1. 启动系统
```bash
# 启动Celery Workers (8个GPU进程)
python start_workers.py

# 启动FastAPI服务
python app.py
```

### 2. 处理视频
```bash
# 使用测试脚本
python test_multi_gpu.py

# 或直接调用API
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{"video_path": "/path/to/video.mp4", "down_sample": true}'
```

### 3. 监控状态
```bash
# 健康检查
curl http://localhost:8000/health

# 任务状态
curl http://localhost:8000/status/{task_id}
```

## 性能优势

1. **并行处理**: 8个GPU同时处理不同帧，理论上8倍加速
2. **内存优化**: 帧级别分发，避免大视频占用过多内存
3. **容错机制**: 单帧处理失败不影响整体处理
4. **负载均衡**: Celery自动分发任务到空闲Worker

## 依赖要求

### 新增依赖
- `fastapi` - Web框架
- `celery` - 分布式任务队列
- `uvicorn` - ASGI服务器

### 保持原有依赖
- `torch_npu` - 华为昇腾NPU支持
- `cv2` - 视频处理
- `PIL` - 图像处理
- `yaml` - 配置文件解析

## 注意事项

1. **内存管理**: 每个Worker占用GPU内存，需确保8个模型能同时加载
2. **任务超时**: 默认5分钟任务超时，可根据视频长度调整
3. **文件路径**: 确保所有Worker进程都能访问输入视频文件
4. **NPU设备**: 适配华为昇腾NPU，通过`ASCEND_RT_VISIBLE_DEVICES`控制设备

## 后续优化建议

1. **Redis后端**: 如需持久化任务状态，可替换为Redis
2. **流式处理**: 支持RTSP实时流处理
3. **动态调度**: 根据GPU负载动态调整任务分配
4. **结果缓存**: 对相同视频片段进行缓存优化