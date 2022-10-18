import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

nclasses = 20 

class Net(nn.Module):
    def __init__(self, type = 'resnet50'):

        super(Net, self).__init__()

        # Resnet50
        if type == 'resnet50':
            self.pretrained_part = models.resnet50(pretrained=True)
            num_ftrs = self.pretrained_part.fc.in_features

            c = 0
            for child in self.pretrained_part.children():
                c+=1
                if c <= 5:
                    child.requires_grad = False

            self.pretrained_part.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Linear(1024, 20))        
            print(self.pretrained_part)
     
        # Resnet50 alternative
        elif type == 'resnet50_2':
            self.pretrained_part =  models.resnet50(pretrained=True)
            num_ftrs = self.pretrained_part.fc.in_features
            c = 0
            for child in self.pretrained_part.children():
                c+=1
                if c <= 4:
                    child.requires_grad = False

            self.pretrained_part.fc = nn.Sequential(nn.Linear(num_ftrs, 1024), nn.ReLU(), nn.Dropout(p = 0.4), nn.Linear(1024, 20))        
            print(self.pretrained_part)
     
        # Resnet152
        elif type == 'resnet152':
            self.pretrained_part = models.resnet152(pretrained=True)
            num_ftrs = self.pretrained_part.fc.in_features

            c = 0
            for child in self.pretrained_part.children():
                c+=1
                if c <= 6:
                    child.requires_grad = False

            self.pretrained_part.fc = nn.Sequential(nn.Linear(num_ftrs, 20))        
            print(self.pretrained_part)

        # Inception
        elif type == 'inception':
            self.pretrained_part = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            
            print(self.pretrained_part)
            c = 0
            
            for child in self.pretrained_part.children():
                print(child)
                c+=1
                if c <= 5:
                    child.requires_grad = False

            n_inputs_fc = self.pretrained_part.fc.in_features
            self.pretrained_part.fc = nn.Sequential(
                nn.Linear(n_inputs_fc, 20),
            )

        # Vgg
        elif type == 'vgg':
            self.pretrained_part = models.vgg16(pretrained = True)
            print(self.pretrained_part)

            for param in self.pretrained_part.parameters():
                param.requires_grad = False
            

            n_inputs = self.pretrained_part.classifier[0].in_features
            self.pretrained_part.classifier = nn.Sequential(
                        # nn.Linear(n_inputs, 2048), 
                        # nn.ReLU(), 
                        # nn.Dropout(0.4),
                        nn.Linear(n_inputs, nclasses))              
            
            self.pretrained_part.classifier.requires_grad = True
        
        # EfficientNet
        elif type == 'efficient':
            self.pretrained_part = models.efficientnet_b0(pretrained=True)

            for param in self.pretrained_part.parameters():
                param.requires_grad = False
            
            n_inputs = list(self.pretrained_part.classifier.children())[-1].in_features
            # n_inputs = 2560

            self.pretrained_part.classifier = nn.Sequential(
                        nn.Linear(n_inputs, 1024), 
                        nn.ReLU(), 
                        nn.Linear(1024, nclasses)) 
            
            self.pretrained_part.classifier.requires_grad = True

        # DenseNet
        elif type == 'densenet':
            self.pretrained_part = models.densenet121(pretrained=True)

            c = 0
            for child in self.pretrained_part.children():
                c+=1
                # if c <= 6:
                print(child)
                child.requires_grad = False

            n_inputs = self.pretrained_part.classifier.in_features
            self.pretrained_part.classifier = nn.Sequential(
                        nn.Linear(n_inputs, nclasses))                   
            
            self.pretrained_part.classifier.requires_grad = True

        else:
            print("Unknown net type")

    def forward(self, x):
        output = self.pretrained_part(x)
        return output
