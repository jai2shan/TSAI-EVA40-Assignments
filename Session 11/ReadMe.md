Q1) Upload the code you used to draw your ZIGZAG or CYCLIC TRIANGLE plot.    
https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2011/draws_this_curve_.ipynb    
Q2) Upload your triangle Plot which was drawn with your code.    
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2011/graph.png    
Q3) Upload the link to your GitHub copy of Colab Code.  
https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2011/Assignment%2011%20Solution.ipynb (Links to an external site.)   

Or

https://colab.research.google.com/github/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2011/Assignment%2011%20Solution.ipynb     
Q4) Upload the github link for the model as described in A11.

https://github.com/jai2shan/TSAI-EVA40-Assignments/blob/master/Session%2011/asgnmt11/assignment11_model.py   
Q5) What is your test accuracy?     
86.54

I tried, Ghost batch normalization, Drop out and GBN with drop out. I am not able to reach 90% accuracy.
Below transformations i tried ,
transforms.Compose([
                                       transforms.RandomRotation(10),
                                       transforms.RandomHorizontalFlip(),  
                                       transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)), 
                                       transforms.RandomCrop(size = 32,padding = 4),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2),
                                       transforms.ToTensor(),
                                       Cutout(n_holes=1, length=8,prob = 0.5),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
                                       
I am not sure where i did wrong. Your comments on that will be very helpful. If possible can you give me a comment where i did wrong? Thank you,Sir
