# Bundle_Recommendation
My UIUC research intern program, still under development.

Due to the volume of train set (about 50G), please contact me if you wanna train on your private development env.

The data format in processed_data will be a list containing:
     -user global id
     -bundle global id
     -edges
     -edgetypes

Train and Test set generation procedure:
     1.using render_dataset.py to get the original version
     2.using negative_sample.py to get the raw neg sample version
     3.using train.py or train_BPR.py to seperate the train and test dataset
     


