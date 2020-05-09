Step - 1
=========
Target: 
-------
1) change the code such that it uses GPU
2) change the architecture to C1C2C3C40 (basically 3 MPs)
3) Params <1M

Results:
--------
1) Params <1M
2) Train Accuracy : 85.74
3) Test Accuracy : 81.78


Step -2 
=======
Target: 
-------
1) Add Depthwise  Depthwise Separable Convolution

Results:
--------
1) Best Train Accuracy : 87.64
2) Test Accuracy : 79.96

Step -3 
=======
Target: 
-------
1) Add Depthwise  Depthwise Separable Convolution and Dilated Convolutions

Results:
--------
1) Best Train Accuracy : 
2) Test Accuracy : 
