# Assignment 8

Trail#1 to 4 
=============
1 - 
Getting the Model Structure correct. Adding basic transformations.
Max Test Accuracy - 81
Result - Model is highly over fitting to train data

2- 
Adding L2 Regularization to reduce over fititng to train
Max Test Accuracy - 80.71
Result - No improvement

3-
Adding GrayScale and Random Crop. Changed learning_rate from 0.001 to 0.01
Max Test Accuracy - 80.4
Result - No improvement

4-
Changed learning_rate from 0.01 to 0.1
Max Test Accuracy - 78.07
Result - No improvement

## Final Solution
Mistake in the top four approaches is that transforms are applied for both train and test
By creating seperate train and test transforms re ran the last script

##### Total Transformations:
transforms.RandomRotation(10),  
transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),  
transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),     
transforms.ToTensor(),   
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))   

##### Added L2 Regularization
Added StepLR after 15 steps

##### Results:
Max Train Accuracy: 92.94
Max Test Accuracy: 87.62

##### Best Model :
Epoch : 15
Train Accuracy :87.45
Test Accuracy : 86.3


