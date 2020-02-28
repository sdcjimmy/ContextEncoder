FROM anibali/pytorch:latest 

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive


RUN sudo apt update ;\
sudo apt install -y s3cmd ;\
sudo apt install -y vim;

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

CMD s3cmd -c /home/user/.s3cfg.cluster sync s3://szu-yeu.hu/SSI.tar.gz ${SLURM_JOB_SCRATCHDIR}/ ;\
tar xvf ${SLURM_JOB_SCRATCHDIR}/SSI.tar.gz --directory ${SLURM_JOB_SCRATCHDIR}/ ;\
python /home/user/train.py -e TEST -cl ;\
s3cmd -c /home/user/.s3cfg.cluster sync ${SLURM_JOB_SCRATCHDIR}/results/ s3://szu-yeu.hu/jobs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}/
