## 容器化启动
启动redis
```
docker run -p 6379:6379 -d --name celery_redis m.daocloud.io/docker.io/redis:6.2.19
```

启动容器（现在只需挂载一个数据目录）
```bash
# 基础启动（返回容器内相对路径）
docker run --privileged --network=host \
    --name videosr \
    --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v $(pwd)/data:/app/data \
    -itd videosr:latest

# 高级启动（返回宿主机绝对路径，便于外部访问）
docker run --privileged --network=host \
    --name videosr \
    --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v $(pwd)/data:/app/data \
    -e DATA_MOUNT_PATH=$(pwd)/data \
    -itd videosr:latest
```

使用curl测试
```bash
# 本地视频文件测试（默认30秒或600帧一个片段）
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "input/videos/example.mp4", "down_sample": true}' > output.txt

# RTSP视频流测试
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "rtsp://username:password@192.168.1.100:554/stream", "down_sample": true}' > output.txt

# 公共RTSP测试流（无需认证）
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{"video_url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4", "down_sample": true}' > output.txt

# 自定义分段设置（15秒或300帧一个片段）
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "input/videos/example.mp4", 
       "down_sample": true,
       "segment_duration": 15.0,
       "segment_max_frames": 300
     }' > output.txt

# RTSP流自定义分段
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "rtsp://192.168.1.100:554/live/stream1", 
       "down_sample": true,
       "segment_duration": 10.0,
       "segment_max_frames": 150
     }' > output.txt
```
测试输出结果会以SSE流的形式输出到output.txt，每个片段处理完成后返回一个JSON结果

### RTSP流地址格式说明
- **有认证**: `rtsp://username:password@ip:port/path`
- **无认证**: `rtsp://ip:port/path`
- **常见端口**: 554（默认）、8554
- **示例厂商格式**:
  - 海康威视: `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101`
  - 大华: `rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0`
  - 通用: `rtsp://ip:port/live/stream1`

### RTSP流处理注意事项
- **网络稳定性**: 确保网络连接稳定，避免流中断
- **认证信息**: 正确配置用户名和密码
- **分段设置**: 
  - 实时处理建议使用较短片段（5-10秒）
  - 网络不稳定时可适当延长片段时长
- **性能考虑**: RTSP流处理会占用更多网络带宽和计算资源
- **流格式**: 支持H.264、H.265等主流编码格式

## 手动启动
### 环境要求

PyTorch >= 2.0.1 CUDA>=12.1 (或华为昇腾NPU环境)

```bash
# 基础依赖
pip install basicsr==1.3.4.9 opencv-python lpips IPython numpy pyyaml

# 多GPU分布式处理依赖
pip install fastapi uvicorn celery redis

# 华为昇腾NPU依赖 (如果使用NPU)
pip install torch-npu
```

### 系统架构

本项目采用分布式架构，支持多GPU并行处理：

- **FastAPI**: Web服务接口
- **Celery**: 分布式任务队列
- **Redis**: 消息代理和结果存储
- **多GPU Workers**: 每个GPU一个独立的Worker进程

### 分段处理模式

视频处理采用分段模式，将长视频分割成小片段进行处理：

- **默认设置**: 每30秒或600帧为一个片段
- **可配置**: 支持自定义片段时长和最大帧数
- **实时返回**: 每个片段处理完成后立即返回结果
- **包含内容**: 
  - 片段的处理前后第一帧对比图（base64编码）
  - 处理前后视频片段的保存路径
  - 片段时间信息和处理统计

#### 环境变量配置

- **`DATA_MOUNT_PATH`**: 可选环境变量，指定挂载目录的绝对路径
  - **未设置时**: 返回容器内相对路径 `data/output_segments/...`
  - **设置后**: 返回宿主机绝对路径 `${DATA_MOUNT_PATH}/output_segments/...`

#### 返回结果格式

**未设置环境变量时（容器内路径）：**
```json
{
  "video_id": "uuid",
  "segment_id": 0,
  "segment_start_time": 0.0,
  "segment_end_time": 30.0,
  "segment_frame_count": 900,
  "before_first_frame": "base64_encoded_image",
  "after_first_frame": "base64_encoded_image", 
  "before_segment_url": "data/output_segments/uuid_segment_0_before.mp4",
  "after_segment_url": "data/output_segments/uuid_segment_0_after.mp4",
  "processing_time": 15.2
}
```

**设置环境变量时（宿主机绝对路径）：**
```json
{
  "video_id": "uuid",
  "segment_id": 0,
  "segment_start_time": 0.0,
  "segment_end_time": 30.0,
  "segment_frame_count": 900,
  "before_first_frame": "base64_encoded_image",
  "after_first_frame": "base64_encoded_image", 
  "before_segment_url": "/path/to/host/data/output_segments/uuid_segment_0_before.mp4",
  "after_segment_url": "/path/to/host/data/output_segments/uuid_segment_0_after.mp4",
  "processing_time": 15.2
}
```

### 多GPU分布式运行说明

#### 1. 启动Redis服务
```bash
# 使用Docker启动Redis (推荐)
docker run -p 6379:6379 -d --name celery_redis redis:6.2.19

# 或使用系统Redis服务
systemctl start redis
```

#### 2. 启动Celery Workers
```bash
# 激活环境
conda activate videosr

# 启动多GPU Workers (自动为每个GPU启动一个Worker)
python start_workers.py
```

#### 3. 启动FastAPI服务
```bash
# 启动Web API服务
python app.py
```

#### 4. 使用方式

**方式一：Web API调用（分段处理模式）**
```bash
# 本地视频文件（默认30秒片段）
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "data/input/video.mp4",
       "down_sample": true
     }'

# RTSP实时视频流
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "rtsp://192.168.1.100:554/live/stream1",
       "down_sample": true
     }'

# 自定义分段设置
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "data/input/video.mp4",
       "down_sample": true,
       "segment_duration": 15.0,
       "segment_max_frames": 300
     }'

# RTSP流自定义短片段（适合实时处理）
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "rtsp://admin:password@192.168.1.100:554/stream",
       "down_sample": true,
       "segment_duration": 5.0,
       "segment_max_frames": 75
     }'
```

**方式二：直接脚本调用**
```bash
python process_video.py -c options/video.yml
```

**方式三：Jupyter Notebook**
```bash
# 直接运行
jupyter notebook process_video.ipynb
```

#### 5. 健康检查
```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查任务状态
curl http://localhost:8000/status/{task_id}
```

### 配置文件示例

```
model_type: HATModel
is_train: false
dist: false
scale: 4
num_gpu: 3  # set num_gpu: 0 for cpu mode
manual_seed: 0

tile:
  tile_size: 512 # max patch size for the tile mode
  tile_pad: 32
  
# network structures
network_g:
  type: HAT
  upscale: 4
  in_chans: 3
  img_size: 64
  window_size: 16
  compress_ratio: 3
  squeeze_factor: 30
  conv_scale: 0.01
  overlap_ratio: 0.5
  img_range: 1.
  depths: [6, 6, 6, 6, 6, 6]
  embed_dim: 180
  num_heads: [6, 6, 6, 6, 6, 6]
  mlp_ratio: 2
  upsampler: 'pixelshuffle'
  resi_connection: '1conv'

# path
path:
  # 预训练模型保存路径
  pretrain_network_g: pretrained_weights/RealHATGAN-TG.pth
  strict_load_g: true
  param_key_g: 'params_ema'

dataset: SingleVideoDataset
# 输入视频目录
video_dir: input/videos/
# 输出视频目录
output_dir: output/

```

### 故障排除

#### 常见问题

1. **NPU多进程错误**
   ```
   Cannot re-initialize NPU in forked subprocess
   ```
   解决方案：已配置使用`solo` worker pool，避免fork问题

2. **Redis连接失败**
   ```
   ConnectionError: Error connecting to Redis
   ```
   解决方案：确保Redis服务正在运行
   ```bash
   docker ps | grep redis
   ```

3. **Worker无法启动**
   ```
   ModuleNotFoundError: No module named 'xxx'
   ```
   解决方案：确保在正确的conda环境中安装依赖
   ```bash
   conda activate videosr
   pip install missing_package
   ```

4. **任务不执行**
   - 检查Workers是否运行：`ps aux | grep celery`
   - 检查Redis连接：`python -c "import redis; print(redis.Redis().ping())"`
   - 检查任务队列：查看Worker日志

5. **路径问题**
   - 容器内相对路径：不设置`DATA_MOUNT_PATH`环境变量
   - 宿主机绝对路径：设置`-e DATA_MOUNT_PATH=$(pwd)/data`

6. **RTSP流问题**
   ```
   无法打开视频流 / 连接超时
   ```
   解决方案：
   - 检查网络连接：`ping 摄像头IP地址`
   - 验证RTSP地址：使用VLC等播放器测试
   - 检查认证信息：确保用户名密码正确
   - 防火墙设置：确保554端口开放
   - 尝试不同的RTSP路径格式

#### 性能优化

- **GPU内存**: 如GPU内存不足，可在`config.py`中调整`tile_size`
- **并发数**: 默认每GPU一个Worker，可根据显存调整
- **批处理**: 大视频会自动分帧并行处理
- **RTSP流优化**:
  - 使用较短的分段时长（5-10秒）以减少延迟
  - 根据网络带宽调整`segment_max_frames`
  - 确保足够的网络带宽避免丢帧
  - 考虑在网络边缘部署以减少传输延迟

### 日志查看

```bash
# 查看Worker日志
tail -f /tmp/celery_worker_gpu*.log

# 查看API服务日志
python app.py  # 直接运行查看控制台输出
```

