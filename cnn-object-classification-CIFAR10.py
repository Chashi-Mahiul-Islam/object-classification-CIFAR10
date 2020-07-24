# CIFAR10 classification

import torch
import numpy as np

# =============================================================================
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else: 
    print('CUDA is available! Training on GPU...')
# =============================================================================

# =============================================================================
# loading data
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

# samples per batch
batch_size = 20
# validation percentage from training data
validation_size = 0.2

# image to nomralized torch.FloatTensor
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
# training and test dataset
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# training indices for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validation_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split] # 0-9999 = validation, 10000-50000 = training

# samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# =============================================================================

# =============================================================================
# visualize
import matplotlib.pyplot as plt

# function to denormalize and display an image
def imshow(img):
    img = img / 2 + 0.5 #denormalize
    plt.imshow(np.transpose(img, (1,2,0))) # convert from tensor image

# one batch of training data
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# imgs in batch with corresponding labels
fig = plt.figure(figsize=(25,4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
# =============================================================================

# =============================================================================
# architecture
from torch import nn, optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        # linear layer 64*4*4 -> 500
        self.fc1 = nn.Linear(64*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
        
        self.dropout = nn.Dropout(p=0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x             
# =============================================================================

# =============================================================================
# create CNN
model =Network()
print(model)


if train_on_gpu:
    model.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# =============================================================================

# =============================================================================
# training
epochs = 30
# keeps minimum validation_loss
valid_loss_min = np.Inf

for epoch in range(epochs):
    
    train_loss = 0
    valid_loss = 0
    
    # training the model
    
    model.train()
    for images, labels in train_loader:
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
            
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
        
    # validating the model 
    model.eval()
    for images, labels in valid_loader:
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
        
        output = model(images)
        
        loss = criterion(output, labels)
        
        valid_loss += loss.item()*images.size(0)
        
    # avg losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    # print losses per epoch 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            (epoch+1), train_loss, valid_loss))
    
    # save model if valid_loss<= valid_lossmin
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}). Saving Model ...'.format(
                valid_loss_min,
                valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss
# =============================================================================
        
# =============================================================================
# load last saved model with minimum valid_loss
model.load_state_dict(torch.load('model_cifar.pt'))

# test
test_loss = 0
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))

model.eval()

for images, labels in test_loader:
    if train_on_gpu:
        images, labels = images.cuda(), labels.cuda()
        
    output = model(images)
    loss = criterion(output, labels)
    test_loss += loss.item()*images.size(0)
    _, pred = torch.max(output, 1)
    equals = pred == labels.view(*pred.shape)
    correct = np.squeeze(equals.numpy()) if not train_on_gpu else np.squeeze(equals.cpu().numpy())
    
    # test accuracy for each object class
    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
# =============================================================================
        

