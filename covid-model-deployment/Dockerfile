FROM python:3.8
WORKDIR /usr/local/

COPY model /usr/local/model
ADD predict.py .
ADD server.py .


RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install tensorflow
RUN pip install numpy
RUN pip install opencv-python
RUN pip install fastapi
RUN pip install starlette
RUN pip install uvicorn
RUN pip install python-multipart

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
