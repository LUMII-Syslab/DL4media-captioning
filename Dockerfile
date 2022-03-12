FROM python:3.7.3-stretch

ARG repo=https://github.com/LUMII-Syslab/DL4media-captioning/releases/download/model
ARG target_dir=/app/model

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app

RUN wget -N $repo/captions_title_LECO2019.json -P $target_dir
RUN wget -N $repo/checkpoint -P $target_dir
RUN wget -N $repo/ckpt-58.data-00000-of-00001 -P $target_dir
RUN wget -N $repo/ckpt-58.index -P $target_dir

# Expose the Flask port
EXPOSE 5000

CMD [ "python", "-u", "./main.py" ]
