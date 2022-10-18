
"""
Resplit data into train and val
Apply on folder with category images
"""

import splitfolders

TRAIN_RATIO = 0.85

splitfolders.ratio("data_temp_for_resize", output="data_crop_2_85", seed=1337, ratio=(TRAIN_RATIO, 1-TRAIN_RATIO), group_prefix=None) # default values
