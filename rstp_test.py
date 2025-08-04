import cv2
import numpy as np
import uvicorn
import torch_npu
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from tgsr.models.hat_model import HATModel
from torchvision.transforms.functional import to_tensor
from PIL import Image
import yaml

app = FastAPI()
@app.on_event("startup")
async def load_model():
    global sr_model
    # 加载配置文件
    config_path = "options/video.yml" 
    with open(config_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    sr_model = HATModel(opt)
    # sr_model.eval()
    print("模型加载完成")



def process_frame(frame: np.ndarray, down_sample=True):
    """处理一帧图像并返回超分结果"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if down_sample:
        print("下采样")
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

    img_pil = Image.fromarray(img)
    tensor = to_tensor(img_pil).unsqueeze(0).to(sr_model.device)

    sr_model.feed_data({'lq': tensor})
    sr_model.pre_process()

    if 'tile' in sr_model.opt:
        sr_model.tile_process()
    else:
        sr_model.process()

    sr_model.post_process()
    result = sr_model.get_current_visuals()['result']
    result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result_np = (result_np * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

def encode_frame(frame: np.ndarray):
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


async def frame_generator(cap, down_sample: bool):
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail="无法打开视频流")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO]视频读取结束")
            break

        print(f"[INFO] 接收到第 {frame_idx} 帧，开始处理...")
        sr_frame = process_frame(frame, down_sample)
        print(f"[INFO] process_frame处理完成")
        img_bytes = encode_frame(sr_frame)
        print(f"[INFO] encode_frame处理完成")
        yield img_bytes
        # yield (
        #     b'--frame\r\n'
        #     b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n'
        # )

        frame_idx += 1

    cap.release()


@app.post("/sr")
async def super_resolution(request: Request):
    try:
        print("进入super_resolution")
        data = await request.json()
        rtsp_url = data["rtsp_url"]
        # cap = cv2.VideoCapture(rtsp_url)
        cap = cv2.VideoCapture("input/videos/example.mp4")
        print("接收视频cap")
        if not cap.isOpened():
            raise HTTPException(status_code=404, detail="视频源无法打开")

        down_sample = data.get("down_sample", True)
        return StreamingResponse(
            frame_generator(cap, down_sample),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)

import asyncio
import os

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)
    # 加载模型
    config_path = "options/video.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    sr_model = HATModel(opt)
    print("模型加载完成")
    

    # 打开视频
    cap = cv2.VideoCapture("input/videos/example.mp4")
    print("接收视频cap")
    if not cap.isOpened():
        print("视频源无法打开")
    else:
        async def run_test():
            frame_idx = 0
            async for chunk in frame_generator(cap, down_sample=True):
                print(f"输出第 {frame_idx} 帧，大小: {len(chunk)}")
                output_path = f"output_frames/frame_{frame_idx:04d}.jpg"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(chunk)
                print(f"已保存为 {output_path}")
                frame_idx += 1

        asyncio.run(run_test())


