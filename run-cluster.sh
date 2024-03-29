#!/bin/bash
#python /home/user/train.py -e cenet-linear-nop-d-hinge1      -m ce-net -n 200 -b 32 -l 0.001   -cd uniform -cl -lx hinge    -g 0 &
#python /home/user/train.py -e cenet-linear-nop-d-hinge2      -m ce-net -n 200 -b 32 -l 0.0001  -cd uniform -cl -lx hinge    -g 1 &
#python /home/user/train.py -e cenet-linear-nop-d-dcgan1      -m ce-net -n 200 -b 32 -l 0.001   -cd uniform -cl -lx dcgan    -g 2 &
#python /home/user/train.py -e cenet-linear-nop-d-dcgan2      -m ce-net -n 200 -b 32 -l 0.0001  -cd uniform -cl -lx dcgan    -g 3 &
#wait

#python /home/user/train.py -e cenet-linear-p-d-hinge1      -m ce-net -n 200 -b 32 -l 0.001   -cd uniform -cl -lx hinge    -p -g 0 &
#python /home/user/train.py -e cenet-linear-p-d-hinge2      -m ce-net -n 200 -b 32 -l 0.0001  -cd uniform -cl -lx hinge    -p -g 1 &
#python /home/user/train.py -e cenet-linear-p-d-dcgan1      -m ce-net -n 200 -b 32 -l 0.001   -cd uniform -cl -lx dcgan    -p -g 2 &
#python /home/user/train.py -e cenet-linear-p-d-dcgan2      -m ce-net -n 200 -b 32 -l 0.0001  -cd uniform -cl -lx dcgan    -p -g 3 &
#wait

#python /home/user/train.py -e vggnet-linear-nop-d-hinge1      -m vgg-unet -n 200 -b 32 -l 0.001   -cd uniform -cl -lx hinge     -g 0 &
#python /home/user/train.py -e vggnet-linear-nop-d-hinge2      -m vgg-unet -n 200 -b 32 -l 0.0001  -cd uniform -cl -lx hinge     -g 1 &
#python /home/user/train.py -e vggnet-linear-nop-d-dcgan1      -m vgg-unet -n 200 -b 32 -l 0.001   -cd uniform -cl -lx dcgan     -g 2 &
#python /home/user/train.py -e vggnet-linear-nop-d-dcgan2      -m vgg-unet -n 200 -b 32 -l 0.0001  -cd uniform -cl -lx dcgan     -g 3 &


#python /home/user/train.py -e cenet-linear-gau-nop-d-hinge-mp75      -m ce-net -n 250 -b 32 -l 0.0001  -cd gaussian -lx hinge -g 0 -mp 0.75 -rs  -cl &
#python /home/user/train.py -e cenet-linear-gau-nop-d-hinge-mp60      -m ce-net -n 250 -b 32 -l 0.0001  -cd gaussian -lx hinge -g 1 -mp 0.6 -rs  -cl &
#python /home/user/train.py -e cenet-linear-uni-nop-d-hinge-mp75      -m ce-net -n 250 -b 32 -l 0.0001  -cd uniform  -lx hinge -g 2 -mp 0.75 -rs  -cl &
#python /home/user/train.py -e cenet-linear-uni-nop-d-hinge-mp60      -m ce-net -n 250 -b 32 -l 0.0001  -cd uniform  -lx hinge -g 3 -mp 0.6 -rs  -cl &


#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp75-1      -m res-ce-net -n 250 -b 32 -l 0.001  -cd gaussian -lx hinge -g 0 -mp 0.75 -rs  -cl &
#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp75-2      -m res-ce-net -n 250 -b 32 -l 0.0005  -cd gaussian -lx hinge -g 1 -mp 0.75 -rs  -cl &
#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp75-3      -m res-ce-net -n 250 -b 32 -l 0.0001  -cd gaussian -lx hinge -g 2 -mp 0.75 -rs  -cl &
#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp75-4      -m res-ce-net -n 250 -b 32 -l 0.00001  -cd gaussian -lx hinge -g 3 -mp 0.75 -rs  -cl &


#python /home/user/train.py -e ceresnet-linear-gau-nop-nod-hinge-mp75-1      -m res-ce-net -n 250 -b 32 -l 0.001  -cd gaussian -lx hinge -g 0 -mp 0.75 -rs  -cl -d &
#python /home/user/train.py -e ceresnet-linear-gau-nop-nod-hinge-mp75-2      -m res-ce-net -n 250 -b 32 -l 0.0005  -cd gaussian -lx hinge -g 1 -mp 0.75 -rs  -cl -d &
#python /home/user/train.py -e ceresnet-linear-gau-nop-nod-hinge-mp75-3      -m res-ce-net -n 250 -b 32 -l 0.0001  -cd gaussian -lx hinge -g 2 -mp 0.75 -rs  -cl -d &
#python /home/user/train.py -e ceresnet-linear-gau-nop-nod-hinge-mp75-4      -m res-ce-net -n 250 -b 32 -l 0.00001  -cd gaussian -lx hinge -g 3 -mp 0.75 -rs  -cl -d &

#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp99      -m res-ce-net -n 250 -b 32 -l 0.001  -cd gaussian -lx hinge -g 0 -mp 0.99  -rs  -cl &
#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp95      -m res-ce-net -n 250 -b 32 -l 0.0005  -cd gaussian -lx hinge -g 1 -mp 0.95 -rs  -cl &
#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp90      -m res-ce-net -n 250 -b 32 -l 0.0005  -cd gaussian -lx hinge -g 1 -mp 0.90 -rs  -cl &
#python /home/user/train.py -e ceresnet-linear-gau-nop-d-hinge-mp50      -m res-ce-net -n 250 -b 32 -l 0.0001  -cd gaussian -lx hinge -g 2 -mp 0.50  -rs  -cl &

python /home/user/train.py -e dicom-resnet-1      -m dicom-resnet -n 250 -b 32 -l 0.001  -cl -g 0 & 
sleep 5 
python /home/user/train.py -e dicom-resnet-2      -m dicom-resnet -n 250 -b 32 -l 0.0001 -cl -g 1 & 
sleep 5
python /home/user/train.py -e dicom-vgg-1         -m dicom-vggnet -n 250 -b 32 -l 0.001  -cl -g 2 & 
sleep 5
python /home/user/train.py -e dicom-vgg-2         -m dicom-vggnet -n 250 -b 32 -l 0.0001 -cl -g 3 & 

wait
touch complete
