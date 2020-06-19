# Tutorial to run the context encdoer project



## Self-supervised learning 

### Step1: Preparing the input and output

#### SSI dataset

All the data location is specifiy a spreadsheet: SSI.csv 

The spreadsheet should contain the following columns:
- `Image_Path`: the absolute image path to the image
- `Study`: the DICOM tag study description
- `Probe`: the DICOM tag probe type
- `Series`: (optional) the DICOM tag study series
- `relative_path`: (optional) the relative image path to the ssi.csv file


**!!IMPORTANT!!**  
Currently the data folder is located at Jimmys workstation
location: */media/jimmy/224CCF8B4CCF57E5/Data/SSI/*

Please copy the data to your own data image folder, and update the Image_Path column in the ssi.csv file. Once you have set the correct image path, you can direct the data loader function to load the images by setting in:

**lib/training/trainer: (line 62-63)**  
self.data_path = *YOUR/PATH/TO/THE/DATA/FOLDER*  
self.img_file = *YOUR/PATH/TO/SSI.CSV/FILE*

#### Output path
You can set the output folder at 

**lib/training/trainer: (line 60)**
self.output_dir = *YOUR/PATH/TO/THE/OUTPUT*

All the output will be saved by the experiment name at the output folder. For example, if you set the output dir at `/home/mywork/results/`, then the output will save in  `/home/mywork/results/EXP-NAME/`

You should expect 3-4 type of files in the folder if run sucessfully - 
1. `exp.ini`: the file save all the training details, like hyperparameters and results
2. `loss_history.csv`: the csv file that save all the training and validation loss
3. `epoch_x.pth`: all the model weights saved during the trianing.
4. `events.out.tfevents.xxx`: the tensorboard files, might not be presented in some training.


### Step2: Train the model

Once you have the SSI dataset ready, you can start training the model. 

```
python train.py -e TEST
```

The parameter `-e` specify the experiment name. The output results will store in the result folder with the experiement name. If the experiement name is `TEST`, the model will use a small subset of the data with only 100 samples to test the model.

For other hyperpameters, please see the **documentation.md** for the parameters.



## Examples
Here's some example experiment used in the paper
- 

##### Example1
```
python train.py -e exp1 -m ce-net -l 0.001 -lx hinge -mp 0.9 -rs
```
- experiment name: exp1
- use context encoder with VGG16 backbone
- use learning rate 0.001
- use hinge loss for GAN training
- MSE loss percentage 90%, Advasarial loss 10%


##### Example2
```
python train.py -e exp2 -m res-ce-net -l 0.01 -lx dcgan -mp 0.75 -d -rs
```
- experiment name: exp2
- use context encoder with RESNET backbone
- use learning rate 0.01
- use dcgan loss for GAN training
- MSE loss percentage 90%, Advasarial loss 10%
- Do not include the DICOM labels for training
- randomly shift the input cropped centers


2. Downstream tasks

#### 


### Train an self-supervised learning model
```
python train.py -e TEST 
```


### Train an downstream tasks

```
python train-downstream.py -e 
```
