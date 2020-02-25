FROM nvidia/cuda:10.1-base-ubuntu16.04

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN apt update ;\
apt install -y s3cmd ;\
apt install -y vim;

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya

# CUDA 10.1-specific steps
RUN conda install "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" -y -c pytorch 
RUN conda install "torchvision=0.5.0=py36_cu101" -y -c pytorch 
RUN conda install cudatoolkit=10.1 -y -c pytorch 
RUN conda install pydicom -c conda-forge
RUN conda install onnx -c conda-forge
RUN conda install scikit-image
RUN conda install scikit-learn
RUN conda install tensorflow
RUN conda install pandas
RUN conda clean -ya

WORKDIR /home/user/
RUN mkdir lib/
RUN mkdir model/
RUN mkdir data/
RUN mkdir weights/
RUN mkdir results/

COPY .s3cfg.cluster .

ADD lib lib/
ADD model model/
ADD train.py .

CMD s3cmd -c /home/user/.s3cfg.cluster sync s3://szu-yeu.hu/SSI/ ${SLURM_JOB_SCRATCHDIR}/
