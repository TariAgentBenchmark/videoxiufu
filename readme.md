## 容器化启动
启动redis
```
docker run -p 6379:6379 -d --name celery_redis m.daocloud.io/docker.io/redis:6.2.19
```

启动容器
```
docker run --privileged --network=host    --name videosr     --device /dev/davinci1     --device /dev/davinci_manager     --device /dev/devmm_svm     --device /dev/hisi_hdc     -v /usr/local/dcmi:/usr/local/dcmi     -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi     -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/     -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info     -v /etc/ascend_install.info:/etc/ascend_install.info -itd videosr:latest
```

使用curl测试
```
curl -X POST "http://localhost:8000/process_video" -H "Content-Type: application/json" -d '{"video_url": "input/videos/example.mp4", "down_sample": true} > output.txt
```
测试输出结果会输出到output.txt

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

**方式一：Web API调用**
```bash
curl -X POST "http://localhost:8000/process_video" \
     -H "Content-Type: application/json" \
     -d '{
       "video_path": "/path/to/input/video.mp4",
       "down_sample": true,
       "output_dir": "output"
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

#### 性能优化

- **GPU内存**: 如GPU内存不足，可在`config.py`中调整`tile_size`
- **并发数**: 默认每GPU一个Worker，可根据显存调整
- **批处理**: 大视频会自动分帧并行处理

### 日志查看

```bash
# 查看Worker日志
tail -f /tmp/celery_worker_gpu*.log

# 查看API服务日志
python app.py  # 直接运行查看控制台输出
```

