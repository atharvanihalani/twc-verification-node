FROM python:3.13-slim
WORKDIR /app
COPY requirements-node.txt .
RUN pip install --no-cache-dir -r requirements-node.txt
COPY twc_common.py collector.py slurm_reporter.py setup.py ./
COPY node ./node
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "node.app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
