# TSAI-EVA40-Assignments
EVA - Assignments
### Assignment 6 - Solution
Link for Jupyter Notebook
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%206/Assignment%20S6%20-%20Solution.ipynb

### 25 misclassified images for "without L1/L2 with BN"
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%206/Images/L1-False%2CL2-False%2CGBN-False.png

### 25 misclassified images for "without L1/L2 with GBN"
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%206/Images/L1-False%2CL2-False%2CGBN-True.png

### Validation Accuracy Change Graph
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%206/Images/MyComp_TestAccuracy.png

### Loss Change Graph
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%206/Images/MyComp_TestLoss.png

### Observations
> L2 regularization is performing significantly better than L1
> Performance Order : L2 Regularization > No Regularization > L1 Regularization
> GBN performs better than BN

### Note
> used only num_splits =1, for GBN, as I am getting error when I tried to give more than 1. Will be helpful if any code can be shared where num_splits is used more than 1 for learning purpose

##### Error Details:
        input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,

    RuntimeError: shape '[-1, 32, 26, 26]' is invalid for input of size 10816
