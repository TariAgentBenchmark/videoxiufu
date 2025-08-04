"""
启动多GPU Celery Workers脚本
为每个GPU设备启动一个Worker进程
"""
import os
import sys
import subprocess
import time
from config import config


def start_worker(device_id: int):
    """
    启动单个GPU Worker
    
    Args:
        device_id: GPU设备ID
    """
    # 设置环境变量
    env = os.environ.copy()
    env.update(config.get_gpu_env_vars(device_id))
    
    # 构建Worker启动命令
    cmd = [
        sys.executable, "-m", "celery",
        "-A", "worker",  # 指向worker模块
        "worker",
        "--loglevel=info",
        f"--hostname=gpu{device_id}@%h",  # 设置唯一的hostname
        "--queues=gpu_queue",  # 指定队列
        "--concurrency=1",  # 每个worker一个进程
    ]
    
    print(f"启动GPU {device_id} Worker...")
    print(f"命令: {' '.join(cmd)}")
    print(f"环境变量: {config.get_gpu_env_vars(device_id)}")
    
    # 启动进程
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process


def main():
    """主函数 - 启动所有GPU Workers"""
    print("=== 多GPU Celery Workers 启动脚本 ===")
    print(f"GPU数量: {config.num_gpus}")
    print(f"GPU设备: {config.gpu_devices}")
    
    # 验证配置
    if not config.validate_config():
        print("配置验证失败，退出")
        sys.exit(1)
    
    # 创建必要目录
    config.setup_directories()
    
    # 启动所有Worker进程
    processes = []
    for device_id in config.gpu_devices:
        try:
            process = start_worker(device_id)
            processes.append((device_id, process))
            time.sleep(2)  # 间隔2秒启动下一个
        except Exception as e:
            print(f"启动GPU {device_id} Worker失败: {e}")
    
    print(f"\n成功启动 {len(processes)} 个Workers")
    print("按 Ctrl+C 停止所有Workers...")
    
    try:
        # 等待所有进程
        for device_id, process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\n正在停止所有Workers...")
        for device_id, process in processes:
            print(f"停止GPU {device_id} Worker...")
            process.terminate()
            process.wait()
        print("所有Workers已停止")


if __name__ == "__main__":
    main()