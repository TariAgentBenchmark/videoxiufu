#!/usr/bin/env python3
"""
测试 Celery worker 是否正常工作
"""
import cv2
import numpy as np
from worker import process_frame_task

def test_single_task():
    """测试单个任务"""
    print("=== 测试 Celery Worker ===")
    
    # 创建测试帧数据
    test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ret, buffer = cv2.imencode('.jpg', test_frame)
    frame_data = buffer.tobytes()
    
    print("发送测试任务...")
    
    try:
        # 发送异步任务
        result = process_frame_task.delay(frame_data, 0, True)
        print(f"任务ID: {result.id}")
        print("等待任务完成...")
        
        # 等待结果（超时30秒）
        task_result = result.get(timeout=30)
        print(f"任务结果: {task_result}")
        
        if task_result["success"]:
            print("✅ 测试成功！Worker 正常工作")
        else:
            print(f"❌ 任务执行失败: {task_result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    test_single_task()