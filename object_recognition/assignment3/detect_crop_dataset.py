import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import skimage.io
import os
import torch
from torchvision.transforms.functional import convert_image_dtype
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

"""
Load FastRCNN network to detect birds on the images and crop a square around them
"""


# Load FastRCNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model = model.eval()


OLD_DATASET_NAME = "bird_dataset"
NEW_DATASET_NAME = "bird_dataset_cropped_2"

if not os.path.exists(NEW_DATASET_NAME):
    os.mkdir(NEW_DATASET_NAME)

train_test_val_dir = ["test_images", "train_images", "val_images"]

# Run through train, test, val
for subset in train_test_val_dir:

    path_subset_old = os.path.join(OLD_DATASET_NAME, subset)
    path_subset_new = os.path.join(NEW_DATASET_NAME, subset)
    os.mkdir(os.path.join(path_subset_new))

    categories = [folder for folder in os.listdir(path_subset_old)]

	# Run through categories 
    for category in categories:
        os.mkdir(os.path.join(path_subset_new, category))

        files = [impath for impath in os.listdir(os.path.join(path_subset_old, category))]

        for f in files:
            # Load image
            path_original_image = os.path.join(path_subset_old, category, f)
            image = read_image(path_original_image)
            image_float = convert_image_dtype(image, dtype=torch.float)

            # Apply net
            results = model([image_float])

            # Get bounding box and crop a surrounding squared zone
            coord = results[0]['boxes'].detach().numpy()[0].astype(int)
            x1,x2, y1, y2 = coord[1],coord[3], coord[0],coord[2]

            w, h = (x2 - x1), (y2-y1)
            w_new =  h_new = max(w,h)

            diff_w = w_new - w
            diff_h = h_new - h

            image = skimage.io.imread(path_original_image)

            x1_new = max(int(x1- diff_w/2), 0) 
            x2_new = min(int(x2 + diff_w/2), image.shape[0])
            y1_new = max(int(y1 - diff_h/2), 0)
            y2_new = min(int(y2 + diff_h/2), image.shape[1])

            cropped = image[x1_new:x2_new, y1_new:y2_new]
            
            # Save new image
            skimage.io.imsave(os.path.join(path_subset_new, category, f), cropped)
            print("Cropped:", path_original_image, "p=",  results[0]['scores'][0].detach().numpy())
