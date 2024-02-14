FROM python:3.10.12

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg

# Update pip
RUN pip install --no-cache-dir --upgrade pip

# Install additional dependencies
RUN apt-get update \
    && pip install --no-cache-dir torch -f https://download.pytorch.org/whl/cu111 \
    && pip install --no-cache-dir git+https://github.com/jhj0517/jhj0517-whisper.git \
    && pip install --no-cache-dir transformers \
    && pip install --no-cache-dir faster-whisper

RUN pip install uvicorn fastapi pydantic motor

# Command to keep the container running indefinitely
#CMD ["sleep", "infinity"]
