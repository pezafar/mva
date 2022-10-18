import main2
import torch
from multiprocessing import Process
from multiprocessing import set_start_method

import sys
sys.setrecursionlimit(10000)

try:
    set_start_method('spawn')
except RuntimeError:
    pass

print("Starting experiments")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device:", device)
args = main2.init_args()


args.action = "train"
args.extract_set = "train"
args.train_data = "bslcp"
args.test_data = "bslcp"
args.num_epochs = 20
args.seed = 0
args.bz = 8




# args.folder = "automation_experiments"
# num_heads_list = [2, 4]
# num_heads_list = [8,16]
# num_encoder_layers_list = [8]
# dim_feedforward_encoder_list = [1024, 2048]
# dropout_list = [0.4]


# args.folder = "automation_experiments_2"
# num_heads_list = [8]
# num_encoder_layers_list = [4,8,12,16]
# dim_feedforward_encoder_list = [1024]
# dropout_list = [0.4, 0.7]

# args.folder = "automation_experiments_3"
# num_heads_list = [8]
# num_encoder_layers_list = [16]
# dim_feedforward_encoder_list = [1024]
# dropout_list = [0.4]
# args.num_epochs = 30

# args.folder = "automation_experiments_4"
# num_heads_list = [8]
# num_encoder_layers_list = [12, 14, 16]
# dim_feedforward_encoder_list = [1024]
# dropout_list = [0.4]
# args.num_epochs = 15
# args.lr = 0.00001
# # args.lr = 0.0001
# args.bz = 16


# args.folder = "automation_experiments_9"
# num_heads_list = [4]
# num_encoder_layers_list = [12]
# dim_feedforward_encoder_list = [1024]
# dropout_list = [0.4]
# args.num_epochs = 15
# # args.lr = 0.001
# args.lr = 0.0001
# args.bz = 16

# args.folder = "automation_experiments_rap"
# num_heads_list = [2, 4, 8, 16]
# num_encoder_layers_list = [8]
# dim_feedforward_encoder_list = [1024]
# dropout_list = [0.4]
# args.num_epochs = 15
# # args.lr = 0.001
# args.lr = 0.0001
# args.bz = 16

args.folder = "automation_experiments_2_layers"
num_heads_list = [4]
num_encoder_layers_list = [2, 4, 8 ]
dim_feedforward_encoder_list = [1024]
dropout_list = [0.4]
args.num_epochs = 15
# args.lr = 0.001
args.lr = 0.0001
args.bz = 16


args.seed = 18


# # Default
# args.num_heads = 8
# args.num_encoder_layers = 8
# args.dim_feedforward_encoder = 2048
# args.dropout = 0.4
processes = []
if __name__ == '__main__':
	for num_heads in num_heads_list:
		for num_encoder_layers in num_encoder_layers_list:
			for dim_feedforward_encoder in dim_feedforward_encoder_list:
				for dropout in dropout_list:
					# Experiment parameters
					print()
					args.num_heads = num_heads
					args.num_encoder_layers = num_encoder_layers
					args.dim_feedforward_encoder = dim_feedforward_encoder
					args.dropout = dropout 

					print(args.num_heads, args.num_encoder_layers, args.dim_feedforward_encoder, args.dropout)

					model_load_dir, model_save_dir, results_save_dir = main2.create_folders(args, type = "New")

					# main2.main(args, device, model_load_dir, model_save_dir, results_save_dir)

					p = Process(target=main2.main, args=(args, device, model_load_dir, model_save_dir, results_save_dir))
					p.start()
					processes.append(p)

			for p in processes:
				p.join()
		

