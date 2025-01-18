FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git git-lfs
RUN git clone https://github.com/xkvma/flowers-rec.git .
RUN pip install --no-cache-dir -r requirements.txt
RUN git lfs pull
EXPOSE 5000

CMD ["python", "app.py"]
