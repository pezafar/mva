import os
import time
import copy
import pickle
import json
from math import ceil
from pathlib import Path
import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, long, optim
from torch.utils.tensorboard import SummaryWriter

from utils import Bar
from utils.viz import viz_results_paper
from utils.averagemeter import AverageMeter
from utils.utils import torch_to_list, get_num_signs
from eval import Metric
import math

class TransformerModelOld(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )

        # No need for embedding, we have dim 1024 features produced by I3D
        self.embedding = nn.Embedding(num_tokens, dim_model)
        # self.embedding = nn.Embedding(3, dim_model)
        
        # Remark: dim_model should be 1024 in this case
        # num_tokens = 2 in our case ?
                 
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        # src = self.embedding(src) * np.sqrt(self.dim_model)
        # # print("Shape target", tgt.shape)
        # # print("Shape input", src.shape)
        # # print()
        tgt = torch.tensor(tgt).to(torch.int64)
        tgt = self.embedding(tgt) * np.sqrt(self.dim_model)
        # src = self.positional_encoder(src)
        # tgt = self.positional_encoder(tgt)
        
        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(2,0,1)
        # tgt = tgt.permute(1,0,2)
        # tgt = tgt[:,:,None]
        tgt = tgt.permute(1,0,2)
        # print("Shape target", tgt.shape)
        # print("Shape input", src.shape)
        # print("Shape mask", tgt_mask.shape)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        # print("Before transformer")
        # transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask)

        transformer_out = self.transformer(src, tgt)
        # print("After transformer")
        out = self.out(transformer_out)

        out = out.permute(1,2,0)
        return out
      
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
class TransformerModel(nn.Module):
    # def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,nlayers: int, dropout: float = 0.5):
    def __init__(self, d_model, nhead, nlayers,dim_feedforward_encoder, dropout):
        ntoken = 2
        # d_model = 1024
        # nhead = 8
        # nlayers = 8
        # dropout = 0.4

        self.d_model = d_model
        
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward_encoder, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
       
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.decoder = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.decoder2 = nn.Linear(d_model, ntoken)
        self.do = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = src.permute(2,0,1)
        src = src * math.sqrt(self.d_model)

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        output = self.act(self.decoder(output))
        output = self.decoder2(self.do(output))
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



def format_predictions_output(predictions: torch.Tensor) -> torch.Tensor:
    ret =  predictions.permute(1,2,0)
    return ret 


class TrainerT:
    """
    Transformer model trainer
    """
    # def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout_p, num_classes, device, weights, save_dir):
    def __init__(self, dim_model, num_heads, num_encoder_layers, dim_feedforward_encoder, dropout, num_classes, device, weights, save_dir):

        # Old
        print("Initializing tranformer model with parameters:")
        print("dim_model :",dim_model)
        print("num_heads :",num_heads)
        print("num_encoder_layers :",num_encoder_layers)
        print("dim_feedforward_encoder :",dim_feedforward_encoder)
        print("dropout :",dropout)
        print()


        self.model = TransformerModel(dim_model, num_heads, num_encoder_layers, dim_feedforward_encoder, dropout).to(device)

        if weights is None:
            self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), ignore_index=-100)

        self.mse = nn.MSELoss(reduction='none')
        self.mse_red = nn.MSELoss(reduction='mean')
        self.sm = nn.Softmax(dim=1)
        self.num_classes = num_classes
        self.writer = SummaryWriter(log_dir=f'{save_dir}/logs')
        self.global_counter = 0
        self.train_result_dict = {}
        self.test_result_dict = {}

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, eval_args, pretrained=''):
        self.model.train()
        self.model.to(device)
        batch_size = 16

        # # load pretrained model
        # if pretrained != '':
        #     pretrained_dict = torch.load(pretrained)
        #     self.model.load_state_dict(pretrained_dict)

        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # optimizer = optim.Adam(self.model.parameters(), lr= learning_rate)
        optimizer = optim.AdamW(self.model.parameters(), lr= learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.9)

        for epoch in range(num_epochs):

            print(optimizer.param_groups[0]['lr'])

            epoch_loss = 0
            end = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            bar = Bar("E%d" % (epoch + 1), max=batch_gen.get_max_index())
            count = 0
            get_metrics_train = Metric('train')
            
            # while count < 1 and batch_gen.has_next():
            count_temp = 0
            while batch_gen.has_next():
                count_temp += 1
                self.global_counter += 1
                batch_input, batch_target, batch_target_eval, mask = batch_gen.next_batch(batch_size)
                # mask = mask.permute(1,0)
                batch_input, batch_target, batch_target_eval, mask = batch_input.to(device), batch_target.to(device), batch_target_eval.to(device), mask.to(device)
                #  PERSO
                optimizer.zero_grad()
                predictions = self.model(batch_input)

                predictions = format_predictions_output(predictions)

                # From here, predictions must be of shape: 8,2,86 (from the original version shapes)
                loss = 0
                # print("nb classes", self.num_classes)
                # loss += self.ce(predictions.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                # loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(predictions[:, :, 1:], dim=1), F.log_softmax(predictions.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
                # print("Ready to compute losses")
                # print(predictions.reshape(-1, 2))
                # print()
                loss += self.ce(predictions.transpose(2, 1).reshape(-1, 2),  batch_target.view(-1))
                loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(predictions[:, :, 1:], dim=1), F.log_softmax(predictions.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # print()
                # print("Real pred")
                # print(torch.max(predictions.data, 2))

                _, predicted = torch.max(predictions.data, 1)
                gt = batch_target
                gt_eval = batch_target_eval

                get_metrics_train.calc_scores_per_batch(predicted, gt, gt_eval, mask)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = "({batch}/{size}) Batch: {bt:.1f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:}".format(
                    batch=count + 1,
                    size=batch_gen.get_max_index() / batch_size,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=datetime.timedelta(seconds=ceil((bar.eta_td/batch_size).total_seconds())),
                    loss=loss.item()
                )
                count += 1
                bar.next()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            get_metrics_train.calc_metrics()
            result_dict = get_metrics_train.save_print_metrics(self.writer, save_dir, epoch, epoch_loss/(len(batch_gen.list_of_examples)/batch_size))
            self.train_result_dict.update(result_dict)

            eval_args[7] = epoch
            eval_args[1] = save_dir + "/epoch-" + str(epoch+1) + ".model"
            self.predict(*eval_args)

            # scheduler.step()
            
        with open(f'{save_dir}/train_results.json', 'w') as fp:
            json.dump(self.train_result_dict, fp, indent=4)
        with open(f'{save_dir}/eval_results.json', 'w') as fp:
            json.dump(self.test_result_dict, fp, indent=4)
        self.writer.close()


    def predict(
            self,
            args,
            model_dir,
            results_dir,
            features_dict,
            gt_dict,
            gt_dict_dil,
            vid_list_file,
            epoch,
            device,
            mode,
            classification_threshold,
            uniform=0,
            save_pslabels=False,
            CP_dict=None,
            ):

        save_score_dict = {}
        metrics_per_signer = {}
        get_metrics_test = Metric(mode)

        self.model.eval()
        with torch.no_grad():
            
            if CP_dict is None:
                self.model.to(device)
                self.model.load_state_dict(torch.load(model_dir))

            epoch_loss = 0
            for vid in tqdm(vid_list_file):
                features = np.swapaxes(features_dict[vid], 0, 1)
                if CP_dict is not None:
                    predicted = torch.tensor(CP_dict[vid]).to(device)
                    pred_prob = CP_dict[vid]
                    gt = torch.tensor(gt_dict[vid]).to(device)
                    gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)
                else:
                    input_x = torch.tensor(features, dtype=torch.float)
                    input_x.unsqueeze_(0)
                    input_x = input_x.to(device)

                    # print(input_x)
                    # print(input_x.shape)

                    # predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                    # predictions = self.model(input_x, torch.ones((1,input_x.shape[2]), device=device))
                    predictions = self.model(input_x)
                    predictions = format_predictions_output (predictions)
                    
                    if self.num_classes == 1:
                        # regression
                        num_iter = 1
                        pred_prob = predictions[-1].squeeze()
                        pred_prob = torch_to_list(pred_prob)
                        predicted = torch.tensor(np.where(np.asarray(pred_prob) > args.classification_threshold, 1, 0)).to(device)
                        
                        gt = torch.tensor(gt_dict[vid]).to(device)
                        gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)

                    else:
                        num_iter = 1
                        pred_prob = torch_to_list(self.sm(predictions))[0][1]
                        predicted = torch.tensor(np.where(np.asarray(pred_prob) > args.classification_threshold, 1, 0)).to(device)
                        gt = torch.tensor(gt_dict[vid]).to(device)
                        gt_eval = torch.tensor(gt_dict_dil[vid]).to(device)

                    if uniform:
                        num_signs = get_num_signs(gt_dict[vid])
                        len_clip = len(gt_dict[vid])
                        predicted = [0]*len_clip
                        dist_uni = len_clip / num_signs
                        for i in range(1, num_signs):
                            predicted[round(i*dist_uni)] = 1
                            predicted[round(i*dist_uni)+1] = 1
                        pred_prob = predicted
                        predicted = torch.tensor(predicted).to(device)

                    if save_pslabels:
                        save_score_dict[vid] = {}
                        save_score_dict[vid]['scores'] = np.asarray(pred_prob)
                        save_score_dict[vid]['preds'] = np.asarray(torch_to_list(predicted))
                        continue
                
                loss = 0
                mask = torch.ones(self.num_classes, np.shape(gt)[0]).to(device)
                # loss for each stage
                # for ix, p in enumerate(predictions):
                #     if self.num_classes == 1:
                #         loss += self.mse_red(p.transpose(2, 1).contiguous().view(-1, self.num_classes).squeeze(), gt.view(-1))
                #     else:

                # loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))
                # loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, 1:])


                predictionsCalc = predictions.permute(1,0,2)
                loss += self.ce(predictionsCalc.reshape(-1, 2),  gt.view(-1))

                epoch_loss += loss.item()


                cut_endpoints = True
                if cut_endpoints:
                    if sum(predicted[-2:]) > 0 and sum(gt_eval[-4:]) == 0:
                        for j in range(len(predicted)-1, 0, -1):
                            if predicted[j] != 0:
                                predicted[j] = 0
                            elif predicted[j] == 0 and j < len(predicted) - 2:
                                break

                    if sum(predicted[:2]) > 0 and sum(gt_eval[:4]) == 0:
                        check = 0
                        for j, item in enumerate(predicted):
                            if item != 0:
                                predicted[j] = 0
                                check = 1
                            elif item == 0 and (j > 2 or check):
                                break

                get_metrics_test.calc_scores_per_batch(predicted.unsqueeze(0), gt.unsqueeze(0), gt_eval.unsqueeze(0))
                
                save_score_dict[vid] = {}
                save_score_dict[vid]['scores'] = np.asarray(pred_prob)
                save_score_dict[vid]['gt'] = torch_to_list(gt)

                if mode == 'test' and args.viz_results:
                    if not isinstance(vid, int):
                        f_name = vid.split('/')[-1].split('.')[0]
                    else:
                        f_name = str(vid)

                    viz_results_paper(
                        gt,
                        torch_to_list(predicted),
                        name=results_dir + "/" + f'{f_name}',
                        pred_prob=pred_prob,
                    )
            
            if save_pslabels:
                PL_labels_dict = {}
                PL_scores_dict = {}
                for vid in vid_list_file:
                    if args.test_data == 'phoenix14':
                        episode = vid.split('.')[0]
                        part = vid.split('.')[1]
                    elif args.test_data == 'bsl1k':
                        episode = vid.split('_')[0]
                        part = vid.split('_')[1]

                    if episode not in PL_labels_dict:
                        PL_labels_dict[episode] = []
                        PL_scores_dict[episode] = []

                    PL_labels_dict[episode].extend(save_score_dict[vid]['preds'])
                    PL_scores_dict[episode].extend(save_score_dict[vid]['scores'])

                for episode in PL_labels_dict.keys():
                    PL_root = str(Path(results_dir).parent).replace(f'exps/results/regression', 'data/pseudo_labels/PL').replace(f'exps/results/classification', f'data/pseudo_labels/PL')
                    # print(f'Save PL to {PL_root}/{episode}')
                    if not os.path.exists(f'{PL_root}/{episode}'):
                        os.makedirs(f'{PL_root}/{episode}')
                        pickle.dump(PL_labels_dict[episode], open(f'{PL_root}/{episode}/preds.pkl', "wb"))
                        pickle.dump(PL_scores_dict[episode], open(f'{PL_root}/{episode}/scores.pkl', "wb"))
                    else:
                        print('PL already exist!!')
                return

            if mode == 'test':
                pickle.dump(save_score_dict, open(f'{results_dir}/scores.pkl', "wb"))

            get_metrics_test.calc_metrics()
            save_dir = results_dir if mode == 'test' else Path(model_dir).parent
            result_dict = get_metrics_test.save_print_metrics(self.writer, save_dir, epoch, epoch_loss/len(vid_list_file))
            self.test_result_dict.update(result_dict)
        
        if mode == 'test':
            with open(f'{results_dir}/eval_results.json', 'w') as fp:
                json.dump(self.test_result_dict, fp, indent=4)
