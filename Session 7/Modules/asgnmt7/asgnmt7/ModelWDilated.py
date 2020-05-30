import torch.nn as nn
import torch.nn.functional as F
dropout_value = 0.1
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # Input Block / C1
        self.convblock1 = nn.Sequential(
                                        ## Convolution 1
                                        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
                                        nn.ReLU(), ## Input = 32, Output = 32
                                        nn.BatchNorm2d(32),
                                        nn.Dropout(dropout_value),
                                        
                                        ## Convolution 2
                                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
                                        nn.ReLU(), ## Input = 32, Output = 32
                                        nn.BatchNorm2d(64),
                                        nn.Dropout(dropout_value),

                                        ## Max Pooling
                                        nn.MaxPool2d(2, 2) ## Input = 32, Output = 16
                                    )
        # Convolution 2 Block - C2
        self.convblock2 = nn.Sequential(
                                        ## Convolution 1
                                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False,dilation = 2),
                                        nn.ReLU(), ## Input = 16, Output = 16
                                        nn.BatchNorm2d(128),
                                        nn.Dropout(dropout_value),
                                        
                                        ## Convolution 2
                                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                                        nn.ReLU(), ## Input = 16, Output = 16
                                        nn.BatchNorm2d(128),
                                        nn.Dropout(dropout_value),
                                        ## Max Pooling
                                        nn.MaxPool2d(2, 2) ## Input = 16, Output = 8
                                    )
        # Convolution 3 Block - C3
        self.convblock3 = nn.Sequential(
                                        ## Convolution 1
                                        nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
                                        nn.ReLU(), ## Input = 8, Output = 8
                                        nn.BatchNorm2d(64),
                                        nn.Dropout(dropout_value),
                                        
                                        ## Convolution 2
                                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
                                        nn.ReLU(), ## Input = 8, Output = 8
                                        nn.BatchNorm2d(128),
                                        nn.Dropout(dropout_value),
                                        ## Max Pooling
                                        nn.MaxPool2d(2, 2) ## Input = 8, Output = 4                                    
                                        )
        # Convolution 4 Block - C4
        self.convblock4 = nn.Sequential(
                                        ## Convolution 1
                                        nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(3, 3), padding=1, bias=False),
                                        nn.ReLU(),## Input = 4, Output = 4     
                                        nn.BatchNorm2d(64),
                                        nn.Dropout(dropout_value),
                                        
                                        ## Convolution 2
                                        nn.Conv2d(in_channels=64, out_channels=128,  kernel_size=(3, 3),padding=1,bias=False),
                                        nn.ReLU(),## Input = 4, Output = 4   

                                    )
        # OUTPUT BLOCK
        self.output = nn.Sequential(
                                    nn.AvgPool2d(kernel_size=4),
                                    nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
                                    )
        
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)              
        x = self.output(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)
        