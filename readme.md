### 环境要求

PyTorch >= 2.0.1 CUDA>=12.1

```
pip install basicsr==1.3.4.9 opencv-python lpips IPython numpy pyyaml
```

###  运行脚本

```
python process_video.py -c options/video.yml
```

或直接运行process_video.ipynb

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

