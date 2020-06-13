
import torch.optim as optim
import torch.nn as nn
import torch

def LR_Max(start,end,iters,mode):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.0001)

	lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
	lr_finder.range_test(trainloader, start_lr=start, end_lr=end,
	                     num_iter=iters, step_mode=mode)
	lr_finder.plot()

	import pandas as pd
	lr_ = pd.DataFrame(lr_finder.history)
	lr_max = lr_[lr_['accuracy']==lr_finder.best_accuracy]['lr']
	return lr_max


def LR_Min(lr_max):
	mins = [lr_max.values[0]/i for i in list(range(5,11))]

	for i in mins:
	  net = Net().to(device)
	  
	  tt = TrainTest()
	  test_acc = 0
	  criterion = nn.CrossEntropyLoss()
	  optimizer = optim.SGD(net.parameters(), lr=i, momentum=0.9,weight_decay = 0.0005)

	  for epoch in range(10):
	      print("EPOCH:", epoch)
	      tt.train_(net, device, trainloader, optimizer, criterion, epoch,L1 = False)
	      acc = tt.test_(net, device, testloader) 
	  lr_min[i] = max(tt.train_acc)
	  del net
	  del tt
	  lr_min = max(lr_min, key=lr_min.get)
	  return lr_min