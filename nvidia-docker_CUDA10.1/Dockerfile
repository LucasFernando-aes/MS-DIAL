FROM nvcr.io/nvidia/cuda:10.1-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

#base packages
RUN apt-get update && apt-get install -y \ 
    vim \
    git

# Install python 3.7
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y

# Add 3.7 to the available alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN update-alternatives --set python /usr/bin/python3.7

# Install pip
RUN apt install python3-pip -y
RUN python -m pip install --upgrade pip

## python3 useful packages
RUN pip install --no-cache-dir \
    torch==1.4.0 \
    torchvision==0.5.0 \
    scipy \
    imageio \
    scikit-image \
    matplotlib \
    seaborn \
    umap-learn \
    scikit-learn \
    tqdm

