import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class TrainTest:
    
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
    

    def train_(self,model, device, train_loader, optimizer, criterion, epoch,L1 = False):
          model.train()
          pbar = tqdm(train_loader)
          correct = 0
          processed = 0
          for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)
        
            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
        
            # Predict
            y_pred = model(data)
        
            outputs = model(data)
            loss = criterion(outputs, target)
            if L1 == True:
                l1_crit = nn.L1Loss(size_average=False)
                reg_loss = 0
                for param in model.parameters():
                    reg_loss += l1_crit(param,target=torch.zeros_like(param))

                factor = 0.0005
                loss += factor * reg_loss
            
            self.train_losses.append(loss)
        
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update pbar-tqdm
            
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
        
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
            
    def test_(self,model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        accT =100. * correct / len(test_loader.dataset)
        return accT