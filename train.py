import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import math
import json
import os, sys, logging, argparse, shutil
from datetime import datetime
from model.data import MutimodalData, collate_pool
from model.model import MultiModalNetwork

parser = argparse.ArgumentParser(description='Multimodal conductivity training')

# General arguments
parser.add_argument('--job_name', default='test', type=str)
parser.add_argument('--model_name', type=str, default='multimodal')
parser.add_argument('--datapath', default='/root/data')
parser.add_argument('--cif_path', default='/root/path_to_cif_files')
parser.add_argument('--jobpath', default='/root/jobs')
parser.add_argument('--disable_ckpt', action='store_true')
parser.add_argument('--fp64', action="store_true")
parser.add_argument('--random_seed', default=42, type=int)
parser.add_argument('--train', default=True, type=bool)
parser.add_argument('--test', default=True, type=bool)

# Training arguments
parser.add_argument('--train_bs', default=128, type=int)
parser.add_argument('--val_bs', default=400, type=int)
parser.add_argument('--test_bs', default=400, type=int)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--optimizer', default='Adamax', type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=1e-1, type=float)
parser.add_argument('--scheduler_gamma', default=0.96, type=float)
parser.add_argument('--rcut', type=float, default=8.0)
parser.add_argument('--scalar_weight', type=bool, default=True)
parser.add_argument('--element_sum', type=bool, default=True)

args = parser.parse_args()

class MultimodalTrainer():
    """
    Basic trainer for multimodal training
    """
    def __init__(self, args):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Initializing.")

        self.args = args

        # Init log dir
        if not os.path.exists(self.args.jobpath):
            os.makedirs(self.args.jobpath, exist_ok=True)
        job_path = os.path.join(self.args.jobpath, self.args.job_name)
        self.job_path = job_path
        if not os.path.exists(job_path):
            os.makedirs(job_path, exist_ok=True)

        # Init device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('device = cuda')
        else:
            self.device = torch.device("cpu")
            print('device = cpu')

        # Init dtype
        if self.args.fp64:
            torch.set_default_dtype(torch.float64)
        else:
            torch.set_default_dtype(torch.float32)
        self.dtype = torch.get_default_dtype()

        # Init random seed
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

        # Init dataset
        
        if self.args.train:
            train_data = MutimodalData(self.args.datapath, torch.device, 'train.csv', self.args.cif_path, train=True)
            self.std = train_data.std
            print ('std of the dataset:', self.std)
            val_data = MutimodalData(self.args.datapath, torch.device, 'val.csv', self.args.cif_path, std=self.std)
            self.train_data_loader = DataLoader(train_data, batch_size = self.args.train_bs, collate_fn=collate_pool)
            self.val_data_loader = DataLoader(val_data, batch_size = self.args.val_bs, collate_fn=collate_pool)
        else:
            self.std = 1.
        if self.args.test:
            test_data = MutimodalData(self.args.datapath, torch.device, 'test.csv', self.args.cif_path, std=self.std)
            self.test_data_loader = DataLoader(test_data, batch_size = self.args.test_bs, collate_fn=collate_pool)

        # Init model
        _, example_structure, _ = train_data[0]
        structure_net_params = {
            "orig_atom_fea_len":train_data.elem_emb_len,
            "nbr_fea_len":example_structure[1][1].shape[-1],
            "atom_fea_len":64,
            "n_conv":3,
            "h_fea_len":64
        }
        composition_net_params = {
            "elem_emb_len":train_data.elem_emb_len,
            "elem_fea_len": 64,
            "n_graph": 3,
            "elem_heads": 3,
            "elem_gate": [128],
            "elem_msg": [128],
            "cry_heads": 3,
            "cry_gate": [128],
            "cry_msg": [128],
        }
        if self.args.scalar_weight:
            weight_net_params = {
                "input_dim":64,
                "output_dim":1,
                "hidden_layer_dims":[64,32]
            }
        else:
            weight_net_params = {
                "input_dim":64,
                "output_dim":64,
                "hidden_layer_dims":[64,32]
            }

        output_net_params = {
            "input_dim":64,
            "output_dim":1,
            "hidden_layer_dims":[64,32]
        }

        self.model = MultiModalNetwork(structure_net_params, composition_net_params, 
                                   weight_net_params, output_net_params, self.args.element_sum)
        
        self.model.to(torch.device)

        # Init optimizer and scheduler
        if self.args.optimizer in ['sgd','SGD']:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer in ['Adam','adam']:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer in ['Adamax','adamax']:
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Supported optimizer: SGD, Adam, Adamax')
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.args.scheduler_gamma)
        
        # Load checkpoint from state dict
        self.logpath = os.path.join(job_path, 'train_log.txt')
        self.start_epoch = 0
        if not self.args.disable_ckpt:
            self.statedict_path = os.path.join(job_path, 'state_dicts')
            if os.path.exists(self.statedict_path):
                shutil.rmtree(self.statedict_path)
            print (self.statedict_path)
            if os.path.exists(self.statedict_path) and len(os.listdir(self.statedict_path)) > 0:
                for i in range(self.args.num_epoch, 0, -1):
                    filename = os.path.join(self.statedict_path, f'epoch_{i}_sd.pt')
                    if os.path.isfile(filename):
                        checkpoint = torch.load(filename)
                        self.model.load_state_dict(checkpoint["model_state_dict"])
                        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        self.start_epoch = checkpoint['epoch']
                        self.model.eval()
                        break
            elif not os.path.exists(self.statedict_path):
                os.mkdir(self.statedict_path)
    
    def print_losses(self, losses, prefix=None):
        if prefix is None:
            message = ''
        else:
            message = prefix
        message = message+str(losses)
        print (message)
        print (message, file=open(self.logpath,'a'))
    
    def get_loss(self, pred, target):
        mse = torch.mean(torch.square(pred - target))
        mae = torch.mean(torch.abs(pred - target))
        loss = mse
        return loss, mse, mae

    def test(self, data, std, best_epoch=0, name='test_results.csv'):
        if not best_epoch:
            best_epoch = self.args.num_epoch
        filename = os.path.join(self.statedict_path, f'epoch_{best_epoch}_sd.pt')
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint['epoch']
        self.model.eval()

        #validation
        val_rmse = []
        val_mae = []
        trues = [] 
        preds = []
        ids = []
        for i, (id, composition_input, structure_input, target) in enumerate(data):
            trues += target[0].cpu().view(-1).tolist()
            ids += id
            structure_presence = target[1]
            pred = self.model(composition_input, structure_input, structure_presence) 
            preds += pred.cpu().view(-1).tolist()
            _, mse, mae = self.get_loss(pred, target[0])
            val_rmse.append(mse.item()*std)
            val_mae.append(mae.item()*std)
        val_rmse = np.sqrt(np.mean(val_rmse))
        val_mae = np.mean(val_mae)
        self.print_losses(val_rmse, prefix=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Test {self.args.model_name} epoch {best_epoch}, RMSE:")
        self.print_losses(val_mae, prefix=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Test {self.args.model_name} epoch {best_epoch}, MAE:")
        results = dict()
        results['id'] = ids
        results['pred'] = (np.array(preds)*std).tolist()
        results['true'] = (np.array(trues)*std).tolist()
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.job_path, name))

    def train(self):
        if self.start_epoch == 0:
            start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training started."
        else:
            start_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training resumed from epoch {self.start_epoch}."
        print(start_message)
        best_mae = 1e10
        best_mae_epoch = 0
        for epoch in range(self.start_epoch, self.args.num_epoch):
            train_rmse = []
            train_mae = []
            # switch to train mode
            self.model.train()
            # train
            for i, (id, composition_input, structure_input, target) in enumerate(self.train_data_loader):
                structure_presence = target[1]
                pred = self.model(composition_input, structure_input, structure_presence)
                loss, mse, mae = self.get_loss(pred, target[0])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_rmse.append(mse.item())
                train_mae.append(mae.item())
            self.scheduler.step()
            train_rmse = np.sqrt(np.mean(train_rmse))*self.std
            train_mae = np.mean(train_mae)*self.std
            self.print_losses(train_rmse, prefix=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training {self.args.model_name} epoch {epoch+1}, RMSE:")
            self.print_losses(train_mae, prefix=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training {self.args.model_name} epoch {epoch+1}, MAE:")

            # switch to evaluate mode
            self.model.eval()
            #validation
            val_rmse = []
            val_mae = []
            for i, (id, composition_input, structure_input, target) in enumerate(self.val_data_loader):
                structure_presence = target[1]
                pred = self.model(composition_input, structure_input, structure_presence)
                loss, mse, mae = self.get_loss(pred, target[0])
                val_rmse.append(mse.item())
                val_mae.append(mae.item())
            val_rmse = np.sqrt(np.mean(val_rmse))*self.std
            val_mae = np.mean(val_mae)*self.std
            self.print_losses(val_rmse, prefix=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Validation {self.args.model_name} epoch {epoch+1}, RMSE:")
            self.print_losses(val_mae, prefix=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Validation {self.args.model_name} epoch {epoch+1}, MAE:")

            if val_mae <  best_mae:
                best_mae= val_mae
                best_mae_epoch = epoch
                print(f'Found best weighted MAE {best_mae:.4f} at epoch {epoch+1}')
                print (f'Found best weighted MAE {best_mae:.4f} at epoch {epoch+1}', file=open(self.logpath,'a'))
            
            # Save ckpts every epoch. Statedict is used for retrain; checkpoint is used for inference
            if not self.args.disable_ckpt:
                statedict_filename = os.path.join(self.statedict_path, f"epoch_{epoch+1}_sd.pt")
                torch.save({
                        'epoch': epoch+1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict()
                        }, statedict_filename)
                print(f'Statedict saved at {statedict_filename}')
                print (f'Statedict saved at {statedict_filename}', file=open(self.logpath,'a'))

        finish_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Training {self.args.model_name} finished."
        finish_message += f" Best weighted MAE {best_mae:.4f} at epoch {best_mae_epoch+1}."
        print (finish_message)
        print (finish_message, file=open(self.logpath,'a'))
        return best_mae_epoch+1, self.std

if __name__ == "__main__":
    trainer = MultimodalTrainer(args)
    best_epoch = trainer.train()
    trainer.test(trainer.test_data_loader, best_epoch=best_epoch)
