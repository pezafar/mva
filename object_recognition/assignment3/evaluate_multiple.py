from torchvision.transforms import functional
from data import data_transforms, data_transforms_test
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch.nn.functional as F
import numpy as np
import torch
from model import Net

"""
Evaluate several datasets simultaneously
Different voting methods tested (majority soft and hard voting, or vote when prefered network is not confident enough)

"""


parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

# Models used for evaluation
models_to_load = [
    {
        "path": "exp_50_09_1/save_model_22.pth",
        "type": "resnet50"
    },
    {
        "path": "exp_fin_dense/keep_model_20_90.pth",
        "type": "densenet"
    },
    {
        "path": "exp_fin_resnet50/keep_model_37_91.666664_fin.pth",
        "type": "resnet50"
    },
    {
        "path": "exp_fin_vgg/keep_model_27_80.0.pth",
        "type": "vgg"
    }
]


args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# use_cuda = False
models_list = []


for model_specs in models_to_load:
    print("Loading", model_specs["path"])
    state_dict = torch.load(
        model_specs["path"], map_location=torch.device('cpu'))
    model = Net(type=model_specs["type"])
    model.load_state_dict(state_dict)
    model.eval()
    models_list.append(model)

print("Models loaded")

if use_cuda:
    print('Using GPU')
    # model.cuda()
else:
    print('Using CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dir = args.data + '/test_images/mistery_category'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms(pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))

        all_outputs = []
        sum = np.zeros(20)
        votes = np.zeros(20).astype(int)
        proba_vote = np.zeros(20)

		# Retrieve model outputs
        for model in models_list:
            output = model(data)
            all_outputs.append(F.softmax(output).detach().numpy()[0])

        for output in all_outputs:
            label = np.argmax(output)
            p = output[label]

			# Vote and probability vote
            votes[label] += 1
            proba_vote[label] += p


		# If first model confidence < 0.9, use majority vote
        if np.max(all_outputs[0]) > 0.9:
            pred = np.argmax(all_outputs[0])
        else:
            pred = np.argmax(votes)


		# Alternative: majority vote, if equality: probability vote 
        # winners = np.flatnonzero(votes == np.max(votes))
        # if winners.shape[0] != 0:
        #     pred = np.argmax(proba_vote)
        # else:
        #     pred = np.argmax(votes)

        output_file.write("%s,%d\n" % (f[:-4], pred))


output_file.close()

print("Succesfully wrote " + args.outfile +
      ', you can upload this file to the kaggle competition website')
