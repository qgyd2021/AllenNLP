# bookworm (glibc 2.36) 避免新版 glibc 拒绝加载 torch 1.12 的 libtorch_cpu.so
FROM python:3.10-slim-bookworm

WORKDIR /code

# patchelf: 清除 libtorch_cpu.so 的可执行栈标记（glibc>=2.41 环境仍需此步）
RUN apt-get update \
    && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && patchelf --clear-execstack /usr/local/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so

COPY . /code

ENV NO_PROXY=localhost,127.0.0.1,0.0.0.0
ENV no_proxy=localhost,127.0.0.1,0.0.0.0

CMD ["python3", "main.py"]
