# Dataset 


# Main Execution files

## Self-supervised learning
train.py: the main script to train the self-supervised learning (Context Encoder)
parameters:
- `e`: the name of the experiment saved in the results folder
- `m`: the model used for training
- `n`: number of epochs
- `b`: batch size
- `l`: learning rate
- `s`: the percentage of validation split
- `o`: optimizer
- `r`: the coefficient for regularizer
- `g`: the gpu used
- `cd`: the initialization of the cropped center images
- `d`: whether to use DICOM loss or not
- `p`: whether to pad the center or not
- `rs`: whether to randomly shift the center cropped center images or not
- `cl`: whether to run on the CCDS super cluster
- `lx`: the loss function of discriminator -- hinge loss or DCGAN
- `mp`: the percentage 


## Downstream tasks
train-downstream.py: the main script to train the downstream task
parameters:
- `e`: the name of the experiment saved in the results folder
- `n`: number of epochs
- `b`: batch size
- `l`: learning rate
- `s`: the percentage of validation split
- `o`: optimizer
- `r`: the coefficient for regularizer
- `g`: the gpu used
- `cd`: the initialization of the cropped center images
- `m`: the model used for training
- `p`: using the default pretrained weights on ImageNet
- `f`: whether to freeze the encoder or not
- `pd`: use the pretrain weights, should pass the path to the weights (.pth files)
- `sk`: shrink the training data (experiment on small data regime)
- `t`: the downstream task to test (hri, nerve, quality or thyroid)



# Library

## dataloading
The classes to load different dataset

#### dataset_loader.py
The dataset function to load the SSI dataset for self-supervised learning

#### hri_dataset_loader.py
The dataset function to load the HRI dataset
#### mapping_dict.py
The mapping dictionary to map the raw DICOM metadata to target encoding

*open_dataset_loader.py was not used in the experiements, the script was adapted from Shuhang's code base*

## evaluation

#### dice_score.py
The function to calculate the dice score

#### evaluate.py
The function to evaluate the model performance

#### predicter.py
The classes to handle the inference, contain:
- NetworkPredicter: The parent class to handle the prediction 
- HRINetworkPredicter: The class to handle prediciton of HRI data(liver and kidney segmentation)
- SingleNetworkPredicter: The class to handle prediction of single-class segmentation(Thyroid nodule, nerve)

## loss_functions
#### dice_loss.py
Dice loss
#### weighted_bce_loss.py
Weighted dice loss
#### gan_loss.py
Loss functions for GAN training, including DCGAN and hinge loss

*ND_Crossentropy.py and TopK_loss.py are not used in the experiments, should be fine to remove them*  


## preprocessing
#### single_transforms.py
preprocessing (torch transformation)for image

#### joint_transforms.py
preprocessing (torchtransformation) for image and mask together


## training
#### trainer.py
The parent class to handle context encoder training 

- instance `info`: All the parameters will store in the variable
- instance `results`: The instance to store the results
- method `load_dataset`: load the data using the SSIDataset class
- method `data_split`: split the dataset into training and validation
- method `get_data_loader`: return the data loader for training and validation
- method `get_optimizer`: return the optimizer
- method `get_network`: return the model, there are 3 choices
    1. ce-net: The context encoder network with VGG16 as backbone
    2. res-ce-net: The context encoder network with RESNET50 as backbone
    3. vgg-unet: The U-Net like context encoder with skip connections, using vgg16 as backbone
- method `get_transform`: return the preprocessing method
- method `get_loss_fx`: return the default loss function
- method `padding_center`: pad the center images with the original surrounding areas
- method `get_disc_input`: process the output of the context encoder. If `padding_center` is specify, it will do the padding; otherwise do nothing
- method `sample_images`: randomly sample images for tensorboard visualization
- method `evaluate`: the function to evalute the model 
- method `train`: the core function to train the model
- method `evaluate_train`: evaulate on the training data, will perform at the end of the training
- method `save_results`: save all the results to the folder by `experiment` name. The parameters will store as `exp.ini`
- method `load_weights`: load the model weights for `self.network`

#### trainer_lp.py
The class to handle context encoder training with linear projection layer (This is the final methods use in the paper). It inherit most of the functions in class `NetworkTrainer`. The overridden methods are the followings:

- method `get_network`: return the same model, but use the linear project layers as the discriminator, which takes the DICOM labels as the input (instead of using it as the labels to calculate the losses)
- method `get_gan_loss`: return the loss function for generator and the discriminator. It takes two different loss type - 
    1. `hinge`: hinge loss
    2. `DCGAN`: binary cross-entropy based loss
- method `evaluate`: the modifed method for evaulation
- method `train`: the modified method for training


#### trainer_seg.py
The classes to handle downstream segmentation tasks training. This file contain 3 different classes
1. SegNetworkTrainer: The parent class to handle the segmentation tasks. Most methods are similar to the ones in the `trainer.py`. The main difference is the method `get_network`, there are multiple options:
    1. `unet`: the vanilla unet
    2. `unet-light`: unet with lesser parameters
    3. `vggnet`: unet with VGG16 as backbone
    4. `res-unet`: unet with RESNET as backbone
    5. `r2u-unet`: recurrent unet
    6. `vgg-ce-unet`: VGG-based unet used to train the context encoder
    7. `res-ce-unet`: RESNET-based unet used to train the context encoder
2. HRISegNetworkTrainer: The class to handle the HRI segmentation task. There are two output classes (liver and kidney), and the DICE were calculated separately

3. SingleSegNetworkTrainer: The class to handle the generic segmentation task with one output class (thyroid or nerve).

#### trainer_class.py
The class to handle downstream classification tasks training
- method `get_network`:
    1. `dicom-resnet`: the network to classify the DICOM labels directly, using RESNET as backbone
    2. `dicom-vggnet`: the ntwork to classify the DICOM labels directly, using VGGNET as backbone

## utilities
*Mostly deprecated function, may not be needed to run the script*



# Model
All the models were defined in this directory

####
