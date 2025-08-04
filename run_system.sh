#!/bin/bash
# 多卡视频处理系统启动脚本

echo "=== 多卡视频超分辨率处理系统 ==="
echo "正在启动系统组件..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装或不在PATH中"
    exit 1
fi

# 创建必要目录
echo "创建目录..."
mkdir -p output temp output_frames

# 启动Celery Workers (后台运行)
echo "启动Celery Workers (8个GPU进程)..."
python start_workers.py &
WORKERS_PID=$!

# 等待Workers启动
echo "等待Workers初始化..."
sleep 10

# 启动FastAPI服务
echo "启动FastAPI服务..."
python app.py &
API_PID=$!

echo "系统启动完成!"
echo "FastAPI服务: http://localhost:8000"
echo "健康检查: http://localhost:8000/health"
echo ""
echo "按任意键停止系统..."
read -n 1

# 停止服务
echo "正在停止系统..."
kill $API_PID 2>/dev/null
kill $WORKERS_PID 2>/dev/null

# 清理后台进程
pkill -f "celery.*worker" 2>/dev/null

echo "系统已停止"