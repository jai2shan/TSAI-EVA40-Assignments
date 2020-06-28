# Assignment 12
## Part - A
Link to Data Splitting into Train and Validation    
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/Scripts/PreparingTrainTest.py
Data : https://github.com/jai2shan/TSAI-EVA40-Assignments/tree/master/Session%2012/Data     
### Step 1 : 
Using ReduceLR on Plateau     
Best Test Accuracy acheived:59.17%

##### Link to Solution:
git: https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/ReduceLROnPlateau.ipynb            
colab: https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/ReduceLROnPlateau.ipynb

### Step 2 :
Using OneCycleLR. LR_max is calculated by choosing the LR which has least loss. LR_max/LR_Min is chosen as 10(In Step 3 I tried to find out LR_Min).         
Best Test Accuracy acheived:51.63%

##### Link to Solution:
git : https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/OneCycleLR.ipynb   
colab : https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/OneCycleLR.ipynb

### Step 3 :
Using OneCycleLR. (Currently running during the time of submission)               
Best Test Accuracy acheived: 40.83(Epoch 10)

##### Link to Solution:
git : https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/OneCycleLR%20-%20Step%203.ipynb      
colab :  https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/OneCycleLR%20-%20Step%203.ipynb    

# Final Verdict:
For now "ReduceLR on Plateau" has the best accuracy. May be after the completion of OnecycleLR Step 3, i might get better results.     
Best Accuracy Achieved: 59.17%

## Links :
git: https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/ReduceLROnPlateau.ipynb            
colab: https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/ReduceLROnPlateau.ipynb

References for mean and sd of Tiny Imagenet         
https://www.kaggle.com/rafazz/tinyimagenet-normalized/version/1


## Part - B
Link to Images : https://github.com/jai2shan/TSAI-EVA40-Assignments/tree/master/Session%2012/Dogs              
Link to JSON : https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/Assignemnt%2012.json          
Link to Notebook/Solution : https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2012/Part%20B.ipynb
