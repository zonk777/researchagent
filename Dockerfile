FROM python:3.13-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip install uv --no-cache-dir

# 复制依赖文件
COPY pyproject.toml uv.lock* ./

# 安装 Python 依赖
RUN uv sync --frozen --no-dev

# 复制源代码
COPY src/ src/
COPY webui.py ./

# 预创建运行时目录
RUN mkdir -p /app/.researchagent/memory

# 暴露 Web UI 端口
EXPOSE 7860

# 默认启动 Web UI
CMD ["uv", "run", "python", "webui.py"]
