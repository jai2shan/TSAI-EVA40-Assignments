import matplotlib.pyplot as plt
import torch
import torchvision

def PlotTrainingGraphs(tt):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(tt.train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(tt.train_acc[4000:])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(tt.test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(tt.test_acc)
    axs[1, 1].set_title("Test Accuracy")


import numpy as np
def ViewModelPerformance(testloader,model,classes,device):
    """5. Test the network on the test data
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    We have trained the network for 2 passes over the training dataset.
    But we need to check if the network has learnt anything at all.
    
    We will check this by predicting the class label that the neural network
    outputs, and checking it against the ground-truth. If the prediction is
    correct, we add the sample to the list of correct predictions.
    
    Okay, first step. Let us display an image from the test set to get familiar.
    """
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    """Okay, now let us see what the neural network thinks these examples above are:"""
    images = images.to(device)
    outputs = model(images).to(device)
    
    """The outputs are energies for the 10 classes.
    Higher the energy for a class, the more the network
    thinks that the image is of the particular class.
    So, let's get the index of the highest energy:
    """
    
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))
    
    """The results seem pretty good.
    
    Let us look at how the network performs on the whole dataset.
    """
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    """That looks waaay better than chance, which is 10% accuracy (randomly picking
    a class out of 10 classes).
    Seems like the network learnt something.
    
    Hmmm, what are the classes that performed well, and the classes that did
    not perform well:
    """
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



def Misclassification(dataset,model,classes,device):
    wrong = []
    for images, labels in dataset:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
          if(len(wrong)<25 and predicted[i]!=labels[i]):
            wrong.append([images[i],predicted[i],labels[i]])

          if(len(wrong)>25):
              break

    fig = plt.figure(figsize = (8,8))
    for i in range(25):
      sub = fig.add_subplot(5, 5, i+1)
      #imshow(misclassified_images[i][0].cpu())
      img = wrong[i][0].cpu()
      img = img * 2 + 0.5 
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg,(1, 2, 0)),interpolation='none')
      sub.set_title("P={}, A={}".format(str(classes[wrong[i][1].data.cpu().numpy()]),str(classes[wrong[i][2].data.cpu().numpy()])))
        
    plt.tight_layout()

