import os
import cv2
import yaml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tgsr.models.hat_model import HATModel
from torchvision.transforms.functional import to_tensor
from PIL import Image
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np

# 初始化 FastAPI 应用
app = FastAPI()

# 挂载静态文件目录供前端访问视频文件
app.mount("/videos", StaticFiles(directory="output"), name="videos")

# 加载模型
@app.on_event("startup")
async def load_model():
    global sr_model
    config_path = "options/video.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    sr_model = HATModel(opt)
    print("模型加载完成")

# 图像处理函数
def process_frame(frame: np.ndarray, down_sample=True):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if down_sample:
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

# 存储处理后视频段路径
processed_video_list = []

@app.post("/segment")
async def process_segment(request: Request):
    global processed_video_list

    try:
        data = await request.json()
        rtsp_url = data["rtsp_url"]
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            raise HTTPException(status_code=404, detail="视频源无法打开")

        segment_id = len(processed_video_list) + 1
        segment_duration = 60  # 秒
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = int(segment_duration * fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        os.makedirs("output", exist_ok=True)

        origin_path = f"output/segment_{segment_id}_origin.mp4"
        processed_path = f"output/segment_{segment_id}_processed.mp4"

        origin_writer = cv2.VideoWriter(origin_path, fourcc, fps, (width, height))
        processed_writer = cv2.VideoWriter(processed_path, fourcc, fps, (width, height))

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            origin_writer.write(frame)
            sr_frame = process_frame(frame)
            processed_writer.write(sr_frame)

        origin_writer.release()
        processed_writer.release()
        cap.release()

        # 更新视频段列表
        processed_video_list.append(processed_path)

        # 构造返回地址
        host = request.client.host
        origin_url = f"http://{host}:8000/videos/segment_{segment_id}_origin.mp4"
        processed_url = f"http://{host}:8000/videos/segment_{segment_id}_processed.mp4"

        merged_url = None
        if len(processed_video_list) % 4 == 0:
            clips = [VideoFileClip(path) for path in processed_video_list[-4:]]
            final_clip = concatenate_videoclips(clips, method="compose")
            merged_path = f"output/merged_{segment_id // 4}.mp4"
            final_clip.write_videofile(merged_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
            merged_url = f"http://{host}:8000/videos/merged_{segment_id // 4}.mp4"

        # 返回 JSON
        return JSONResponse(content={
            "origin_video_url": origin_url,
            "processed_video_url": processed_url,
            "merged_video_url": merged_url  # 若非4的倍数，则返回null
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动服务（用于开发测试）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
