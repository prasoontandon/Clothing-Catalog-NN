import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    if training:
        train_set=datasets.FashionMNIST('./data',train=True,
                                       download=True,transform=custom_transform)
        return torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    else:
        test_set=datasets.FashionMNIST('./data',train=False,
                                       transform=custom_transform)       
        return torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=64)

def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    # 4 layers of our Neural Network
    model = nn.Sequential(
        nn.Flatten(), #1
        nn.Linear(28*28, 128), #2
        nn.ReLU(),
        nn.Linear(128, 64), #3
        nn.ReLU(),
        nn.Linear(64,10) #4
    )

    return model

def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   
    model.train()
    for epoch in range(T):
        running_loss = 0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            images, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            total += images.size(0)
            _, predicted = torch.max(outputs, axis=1)
            correct += (predicted == labels).int().sum()
            running_loss += loss.item() * images.size(0)
        
        # print statistics
        print(f'Train Epoch: {epoch}\tAccuracy: {correct}/{total} ({100 * correct / total:.2f}%)\tLoss: {running_loss / total:.3f}')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    #Turn the model into evaluation mode
    model.eval()

    running_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data

            outputs = model(images)
            loss = criterion(outputs, labels)

            total += images.size(0)
            _, predicted = torch.max(outputs, axis=1)
            correct += (predicted == labels).int().sum()
            running_loss += loss.item() * images.size(0)
        
        # print statistics
        if show_loss: print(f'Average loss: {running_loss / total:.4f}')
        print(f'Accuracy: {100 * correct / total:.2f}%')

def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    outputs = model(test_images[index])
    prob = F.softmax(outputs, dim=1)

    #Get the top three values in descending order
    top_k = torch.topk(prob, 3, 1)
    vals = top_k[0][0]
    indx = top_k[1][0]   

    for i,j in zip(indx, vals):
        print(f'{class_names[i]}: {100*j:.2f}%')


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to examine the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()

    train_loader = get_data_loader()
    #print(type(train_loader))
    #print(train_loader.dataset)
    test_loader = get_data_loader(False)
    #print(type(test_loader))
    #print(test_loader.dataset)

    model = build_model()
    #print(model)

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss=False)
    #evaluate_model(model, test_loader, criterion, show_loss=True)

    test_images, test_labels = next(iter(test_loader))
    predict_label(model, test_images, 1)