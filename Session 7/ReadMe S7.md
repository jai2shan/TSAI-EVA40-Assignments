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


Receptive Field:
----------------
Block,Layer,Channel_in,Padding,Kernel,Stride,Channel_out,Receptive field,Jump     
Input,,,,,,32,1,1     
conv - B1,conv 1-1,32,1,3,1,32,3,1      
conv - B1,conv 1-2,32,1,3,1,32,5,1      
conv - B1,MaxPool1,32,0,2,2,16,6,1        
conv - B2,conv 2-1,16,1,3,1,16,14,4       
conv - B2,conv 2-2,16,1,3,1,16,16,1       
conv - B2,MaxPool2,16,0,2,2,8,17,1        
conv - B3,conv 3-1,8,1,3,1,8,21,2       
conv - B3,conv 3-2,8,1,3,1,8,23,1       
conv - B3,MaxPool2,8,0,2,2,4,24,1       
conv - B3,conv 4-1,4,1,3,1,4,28,2     
conv - B3,conv 4-2,4,1,3,1,4,30,1       

Unable to calculate the receptive field values when I am using dilated convolutions. I am aware the dilated convolutions with alpha value 2 will increase receptive field by 4 but not able to implement in the table form to calculate receptive field

Exploration about calculation:
https://distill.pub/2019/computing-receptive-fields/
https://stats.stackexchange.com/questions/265462/whats-the-receptive-field-of-a-stack-of-dilated-convolutions
