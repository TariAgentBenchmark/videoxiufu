# 使用指定的基础镜像
FROM ascendai/pytorch:ubuntu-python3.8-cann8.0.rc1.beta1-pytorch2.1.0

# 设置工作目录
WORKDIR /app

# 更新apt源为清华镜像源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# 更新软件包列表并安装libgomp1
RUN apt-get update && \
    apt-get install -y libgomp1

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
COPY torch-2.4.1-cp38-cp38-manylinux2014_aarch64.whl /app/
RUN pip install torch-2.4.1-cp38-cp38-manylinux2014_aarch64.whl
RUN pip install --verbose basicsr==1.4.2

RUN pip install torch-npu
RUN pip install opencv-python lpips IPython numpy pyyaml

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install einops 

RUN sed -i 's/_functional_tensor/functional_tensor/g' "/usr/local/python3.8/lib/python3.8/site-packages/basicsr/data/degradations.py"
COPY patch.py /app/
RUN python patch.py


COPY torch-2.4.0-cp38-cp38-manylinux2014_aarch64.whl /app/
COPY torch_npu-2.4.0.post3-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl /app/
RUN pip install torch-2.4.0-cp38-cp38-manylinux2014_aarch64.whl
RUN pip install torch_npu-2.4.0.post3-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 设置容器启动时的默认命令
CMD ["bash"]
