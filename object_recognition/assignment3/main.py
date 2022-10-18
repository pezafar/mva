from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.nn.modules.activation import Softmax
from model import Net
from data import data_transforms
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models
from torch.autograd import Variable
from tqdm import tqdm
# custom
from custom_folder import ImageFolderWithPaths

torch.cuda.empty_cache()


# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
# Custom
parser.add_argument('--load', type=str, default='no',
                    help='Load the model from save or not.')


args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

if args.load == None:
    print("No load arg input")
    exit()
else:
    print("Ready to load from experiment",
          args.experiment, "model", int(args.load))


# Create experiment folder
if not os.path.isdir(args.experiment):
    os.makedirs(args.experiment)

# Data initialization and loading
train_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    ImageFolderWithPaths(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)



# model = Net(type='resnet50')
# model = Net(type='densenet')
# model = Net(type='resnet50_2')
# model = Net(type='resnet152')
# model  = Net(type = 'vgg')
model  = Net(type = 'efficient')


# Different optimizers and schedulers tested during the experimentation

# optimizer = optim.Adam(model.parameters(), lr=0.0005)
# optimizer = optim.Adam(model.parameters(), lr=0.00005)
# scheduler = ExponentialLR(optimizer, gamma=0.6)
# scheduler = ReduceLROnPlateau(optimizer, 'min',patience =2)

#Resnet152
# optimizer = optim.Adam(model.parameters(), lr=0.00005)
# scheduler = ExponentialLR(optimizer, gamma=0.5)
# scheduler = ReduceLROnPlateau(optimizer, 'min')

#Inception
# model = Net(type='inception')
# optimizer = optim.Adam(model.parameters(), lr=0.0005)
# scheduler = ExponentialLR(optimizer, gamma=0.6)

# #Vgg
# optimizer = optim.SGD(model.parameters(), lr = 0.01)
# scheduler = ReduceLROnPlateau(optimizer, 'min',patience =2)


#Efficient
# optimizer = optim.Adam(model.parameters(), lr=0.05)
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, 'min',patience =2, factor = 0.1)

#Dense
# optimizer = optim.SGD(model.parameters(), lr = 0.01)
# scheduler = ReduceLROnPlateau(optimizer, 'min',patience =2)



if use_cuda:
    print('Using GPU')
    model.cuda()
    print("Switch to cuda done")
else:
    print('Using CPU')


# Helpers for debug and experimenting 
train_losses = []
val_losses = []

# Ratio of correct predictios in validation
ratios = []


def train(epoch):
    model.train()
    loss_batch = 0
    for batch_idx, (data, target, path) in enumerate(train_loader):

        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        # output = model(data).logits
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_batch += loss.data.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx *
                len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

    # scheduler.step()
    train_losses.append(loss_batch)


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target, paths in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]

        # Print misclassified images paths
        comparison = pred.eq(target.data.view_as(pred)
                             ).detach().reshape((-1, 1))

        # Print examples with wrong outputs
        for i in range(comparison.shape[0]):
            if not comparison[i]:
                print(paths[i])

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    val_losses.append(validation_loss)
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


    # Append precision
    ratios.append(correct / len(val_loader.dataset))
    return validation_loss, 100. * correct / len(val_loader.dataset)

if (args.load != "0"):
    state_dict = torch.load(args.experiment + "/model_" + args.load + ".pth")
    model.load_state_dict(state_dict)

for epoch in range(int(args.load) + 1, args.epochs + 1):
    train(epoch)
    val_loss, correct_percentage = validation()

	# Accuracy plateau detection
    # print("Ratios:", ratios)
    # if len(ratios) >= 2 and (ratios[-1] - ratios[-2]) < 0.05:
    #     scheduler.step()
    #     print()
    #     print("Scheduler step done, lr =", optimizer.param_groups[0]['lr'])
    #     print()


    scheduler.step(val_loss)
    # scheduler.step()
    print("Lr =", optimizer.param_groups[0]['lr'])
    print(correct_percentage)
    model_file = args.experiment + '/model_' + str(epoch) + '_' + str(correct_percentage.numpy()) + '.pth'

    print(train_losses)
    print(val_losses)

    torch.save(model.state_dict(), model_file)
    print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' +
          model_file + '` to generate the Kaggle formatted csv file\n')
