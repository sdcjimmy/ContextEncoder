#!/bin/bash

until [ -f /home/user/complete ]
do
echo Upload tensorboard info!!
sleep 600
s3cmd -c /home/user/.s3cfg.cluster sync ${SLURM_JOB_SCRATCHDIR}/results/ s3://szu-yeu.hu/jobs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}/ --rexclude='.*' --rinclude='./events*'
s3cmd -c /home/user/.s3cfg.cluster sync ${SLURM_JOB_SCRATCHDIR}/results/ s3://szu-yeu.hu/jobs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}/ --rexclude='.*' --rinclude='./exp.ini'
done

