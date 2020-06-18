# Tutorial to run the context encdoer project



## Self-supervised learning 

### Step1: Preparing the data

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

**lib/training/trainer:**  
self.data_path = *YOUR/PATH/TO/THE/DATA/FOLDER*  
self.img_file = *YOUR/PATH/TO/SSI.CSV/FILE*

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
