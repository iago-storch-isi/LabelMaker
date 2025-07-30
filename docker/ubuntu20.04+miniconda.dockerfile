FROM ubuntu:20.04
WORKDIR /
ENV TZ=Europe/Zurich
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone
RUN apt-get update && \
    apt-get -y install git curl wget make nano ffmpeg libsm6 libxext6 unzip && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /Miniconda3-latest-Linux-x86_64.sh && \
    /Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm -rf /Miniconda3-latest-Linux-x86_64.sh && \
    /miniconda3/bin/conda init bash && \
    chmod -R 777 /miniconda3
RUN export PATH="/miniconda3/bin:$PATH" && conda config --set auto_activate_base false

RUN export PATH="/miniconda3/bin:$PATH" && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN export PATH="/miniconda3/bin:$PATH" && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN git clone https://github.com/iago-storch-isi/LabelMaker.git
WORKDIR /LabelMaker
RUN git checkout v2_yuchi

RUN export PATH="/miniconda3/bin:$PATH" && \
    bash env_v2/install_labelmaker_env.sh 3.10 11.8 2.0.0 10.4.0 && \
    rm -rf /root/.cache/* && \
    chmod -R 777 /miniconda3/envs/labelmaker
RUN export PATH="/miniconda3/bin:$PATH" && \
    bash env_v2/install_sdfstudio_env.sh 3.10 11.3 && \
    rm -rf /root/.cache/* && \
    chmod -R 777 /miniconda3/envs/sdfstudio
