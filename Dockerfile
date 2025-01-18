FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/xkvma/flowers-rec.git . -b develop
# Optionally, if there's a requirements.txt for Python dependencies, install them
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
