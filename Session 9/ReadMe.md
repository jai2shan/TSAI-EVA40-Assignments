 Assignment 9

Runs
=============
### 1 - 
Step 1    
Include Horizontalflip in transforms             
Include VerticalFlip in transforms          
Changed StepLR to 20 steps     

Max Test Accuracy - 86.41    

Step 2    
Removed VerticalFlip in transforms     
Max test Accuracy - 89.07    
Max Train Accuracy - 95.07    

##### Best Model Accuracies    
Train - 91.74     
Test - 88.43    


### 2 -
Including GradCam code

Max test Accuracy - 88.25    
Max Train Accuracy - 94.71  


Tried below sources, but was not able to make it work,       
Source : https://github.com/jacobgil/pytorch-grad-cam    
https://github.com/kazuto1011/grad-cam-pytorch    

Finally below Source was working     
https://github.com/Sushmitha-Katti/EVA-4/tree/master/Session9/Template


## Solution
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%209/Assignment_9_Step%202.ipynb      
Or      
https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%209/Assignment_9_Step%202.ipynb     

.py files location 
https://github.com/jai2shan/TSAI-EVA40-Assignments/tree/master/Session%209/packaging/asgnmt9      

quizDNN.py link   
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%209/packaging/asgnmt9/quizDNN.py



**During Assignment submission i noticed that i used Horizontalflip twice in the transformations. Hence reran the script, which took more than 15 mins to run. I am submitting another solution below, can you please consider the latest one.
In this run, I am able to achieve,
Max Test Accuracy - 90.85
Max Train Accuracy - 95.77
Best Model Train and Test Accuracies - 92.90 and 90.03
