# 使用指定的基础镜像
FROM videosr:tari

RUN pip3 install celery==5.5.3 redis==6.2.* -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置工作目录
WORKDIR /app

ADD . .

ENTRYPOINT [ "bash", "run_system.sh" ]