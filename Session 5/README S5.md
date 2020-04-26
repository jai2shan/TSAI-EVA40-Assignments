# TSAI-EVA40-Assignments
EVA - Assignments

## Session 5

### Assignment 5 - Trail 1
Target for Assignment
99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement) Less than or equal to 15 epochs
Less than 10000 Parameters

Target for Trial 1
1) Reduce the number of parameters by changing the model skeleton and bring back the number of variables to less than 10000
2) Reduce the number of epochs and observe how the performance is dropping

Results
Best accuracy (Train) - 98.85%
Best accuracy (Test) - 99.26%
Total Parameters - 9752

Analysis
1) Variables got reduced with drop in accuracy in both train and test data.
2) Model is underfitting

Link to the File : https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%205/EVA4S5F10%20-%20Step%201%20-%20Reduce%20Number%20of%20Variables.ipynb

### Assignment 5 - Trail 2
Target for Assignment
99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement) Less than or equal to 15 epochs
Less than 10000 Parameters

Target for Trial 2
1) Changing the learning rate to 0.025 so that we can hit the max accuracy in the early stages of epochs
2) Trying 0.02,0.025 and 0.03 to choose the best parameter

Results
Best accuracy (Train) - 98.97%
Best accuracy (Test) - 99.47%
Total Parameters - 9752

Analysis
0.03 is giving better performance on observing the Test Accuracy graph we can observe the max point of the accuracy is hitting around epoch 10, if that can be achieved around 6th epoch, LR scheduler can make the test accuracy stable which can be tried in next attempt

Colab Link: https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%205/EVA4S5F10%20-%20Step%202%20-%20Changing%20the%20Learning%20Rate.ipynb

### Assignment 5 - Trail 3   
Target for Assignment
99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
Less than or equal to 15 epochs      
Less than 10000 Parameters

Target for Trial 3
 1) Changing the learning rate to 0.05 so that we can hit the max accuracy in the early stages of epochs   
 2) Trying stepsize of 6,8 and 10 to choose the best parameter

Results
Best accuracy (Train) - 98.97%    
Best accuracy (Test) - 99.47%    
Total Parameters - 9752

Analysis
step size of 6 is giving better performance on observing the Test Accuracy graph.

Colab Link : https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%205/EVA4S5F10%20-%20Step%203%20-%20Changing%20the%20Learning%20Rate%20with%20different%20step%20sizes.ipynb
