FROM python:3.9.16-slim

COPY requirements.txt /config/requirements.txt 
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /config/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./check_job_result.py /app/check_job_result.py

WORKDIR /app

CMD ["python", "/app/check_job_result.py"]
