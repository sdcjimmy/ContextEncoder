#python train.py -e baseline1 -n 100 -b 8 -l 0.001
#python train.py -e baseline2 -n 100 -b 8 -l 0.0001
#python train.py -e baseline3 -n 100 -b 8 -l 0.00001


#python train.py -e baseline4 -n 300 -b 8 -l 0.0001
#python train.py -e baseline5 -n 300 -b 8 -l 0.00005
#python train.py -e baseline6 -n 300 -b 8 -l 0.00001

#python train.py -e base-padding-dcm   -n 150 -b 8 -l 0.0001 -p
#python train.py -e base-padding-nodcm -n 150 -b 8 -l 0.0001 -d -p
#python train.py -e base-nodcm         -n 150 -b 8 -l 0.0001 -d 


python train.py -e base-padding-dcm-uni   -n 100 -b 8 -l 0.0001 -p     -cd uniform
python train.py -e base-padding-nodcm-uni -n 100 -b 8 -l 0.0001 -d -p  -cd uniform
