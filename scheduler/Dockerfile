FROM python:3.12-slim

WORKDIR /app

COPY scheduler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/scheduler

RUN adduser --disabled-password --gecos "" lumina
USER lumina

CMD ["python", "-m", "scheduler.main"]
