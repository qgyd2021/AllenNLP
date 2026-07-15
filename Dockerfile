# bookworm (glibc 2.36) 可正常加载 torch 1.12 的 libtorch_cpu.so；
# 勿用 python:3.10 默认标签（可能指向 trixie / glibc>=2.41）。
FROM python:3.10-slim-bookworm

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir -r /code/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -c "import torch; print('torch', torch.__version__)"

COPY . /code

ENV NO_PROXY=localhost,127.0.0.1,0.0.0.0
ENV no_proxy=localhost,127.0.0.1,0.0.0.0

CMD ["python3", "main.py"]
