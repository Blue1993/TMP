import os
import sys
import yaml
import pickle
import random
import logging
import copy
import math
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from common.utils import load_cfg
from interaction_modules.trainers import BATrainer
from interaction_modules.models import PretrainedBA
from interaction_modules.loaders import TrainDataset, pad_data

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, RandomSampler

def reset_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  

def main():
    
    ##############
    # Load config
    ##############
    print("Run PDBbind training")
    conf_path = "BA_predictor_configuration.yml"
    
    config = load_cfg(conf_path)    
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu") 
    
    ############
    # Load Data
    ############
    with open(config["Path"]["Kfold"], "rb") as f:
        Kfold_index_dict = pickle.load(f)

    Training_df = pd.read_csv(config["Path"]["training_df"], sep = "\t")
    print(f"[Training] number of complexes: {len(Training_df)}")
    PDB_IDs, Uniprot_IDs, Lig_codes, BA_labels = Training_df.iloc[:, 0].values, Training_df.iloc[:, 1].values, Training_df.iloc[:, 2].values, Training_df.iloc[:, 3].values
    Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(PDB_IDs, Uniprot_IDs, Lig_codes)]) 

    CASF2016_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2016_DTA_data.tsv", sep = "\t")
    print(f"[CASF2016] number of complexes: {len(CASF2016_df)}")
    CASF2016_PDB_IDs, CASF2016_Uniprot_IDs, CASF2016_Lig_codes, CASF2016_BA_labels = CASF2016_df.iloc[:, 0].values, CASF2016_df.iloc[:, 1].values, CASF2016_df.iloc[:, 2].values, CASF2016_df.iloc[:, 3].values
    CASF2016_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CASF2016_PDB_IDs, CASF2016_Uniprot_IDs, CASF2016_Lig_codes)])

    CASF2013_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2013_DTA_data.tsv", sep = "\t")
    print(f"[CASF2013] number of complexes: {len(CASF2013_df)}")
    CASF2013_PDB_IDs, CASF2013_Uniprot_IDs, CASF2013_Lig_codes, CASF2013_BA_labels = CASF2013_df.iloc[:, 0].values, CASF2013_df.iloc[:, 1].values, CASF2013_df.iloc[:, 2].values, CASF2013_df.iloc[:, 3].values
    CASF2013_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CASF2013_PDB_IDs, CASF2013_Uniprot_IDs, CASF2013_Lig_codes)])

    CSAR2014_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2014_DTA_data.tsv", sep = "\t")
    print(f"[CSAR2014] number of complexes: {len(CSAR2014_df)}")
    CSAR2014_PDB_IDs, CSAR2014_Uniprot_IDs, CSAR2014_Lig_codes, CSAR2014_BA_labels = CSAR2014_df.iloc[:, 0].values, CSAR2014_df.iloc[:, 1].values, CSAR2014_df.iloc[:, 2].values, CSAR2014_df.iloc[:, 3].values
    CSAR2014_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSAR2014_PDB_IDs, CSAR2014_Uniprot_IDs, CSAR2014_Lig_codes)])

    CSAR2012_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2012_DTA_data.tsv", sep = "\t")
    print(f"[CSAR2012] number of complexes: {len(CSAR2012_df)}")
    CSAR2012_PDB_IDs, CSAR2012_Uniprot_IDs, CSAR2012_Lig_codes, CSAR2012_BA_labels = CSAR2012_df.iloc[:, 0].values, CSAR2012_df.iloc[:, 1].values, CSAR2012_df.iloc[:, 2].values, CSAR2012_df.iloc[:, 3].values
    CSAR2012_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSAR2012_PDB_IDs, CSAR2012_Uniprot_IDs, CSAR2012_Lig_codes)])

    CSARset1_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset1_DTA_data.tsv", sep = "\t")
    print(f"[CSARset1] number of complexes: {len(CSARset1_df)}")
    CSARset1_PDB_IDs, CSARset1_Uniprot_IDs, CSARset1_Lig_codes, CSARset1_BA_labels = CSARset1_df.iloc[:, 0].values, CSARset1_df.iloc[:, 1].values, CSARset1_df.iloc[:, 2].values, CSARset1_df.iloc[:, 3].values
    CSARset1_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSARset1_PDB_IDs, CSARset1_Uniprot_IDs, CSARset1_Lig_codes)])

    CSARset2_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset2_DTA_data.tsv", sep = "\t")
    print(f"[CSARset2] number of complexes: {len(CSARset2_df)}")
    CSARset2_PDB_IDs, CSARset2_Uniprot_IDs, CSARset2_Lig_codes, CSARset2_BA_labels = CSARset2_df.iloc[:, 0].values, CSARset2_df.iloc[:, 1].values, CSARset2_df.iloc[:, 2].values, CSARset2_df.iloc[:, 3].values
    CSARset2_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(CSARset2_PDB_IDs, CSARset2_Uniprot_IDs, CSARset2_Lig_codes)])

    Astex_df = pd.read_csv("/home/ssm/data/work/Pseq2affinity/data/PDBbind/Astex_DTA_data.tsv", sep = "\t")
    print(f"[Astex] number of complexes: {len(Astex_df)}")
    Astex_PDB_IDs, Astex_Uniprot_IDs, Astex_Lig_codes, Astex_BA_labels = Astex_df.iloc[:, 0].values, Astex_df.iloc[:, 1].values, Astex_df.iloc[:, 2].values, Astex_df.iloc[:, 3].values
    Astex_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(Astex_PDB_IDs, Astex_Uniprot_IDs, Astex_Lig_codes)])

    #######################
    # CV setting and Train
    ####################### 
    #for idx in range(5):
    for idx in [1, 0, 4, 2, 3]:
        reset_seed(0)
        
        DTADataset = TrainDataset(interaction_IDs = Interactions_IDs, labels = BA_labels, device = device,
                                        protein_features_path = config["Path"]["training_protein_feat"],
                                        pocket_indices_path = config["Path"]["training_pocket_ind"],
                                        compound_features_path = config["Path"]["training_ligand_graph"],
                                        interaction_sites_path = config["Path"]["training_interaction"],)
                                        
        print(f">>> CV {idx} is running ...")
        train_index, val_index, test_index = Kfold_index_dict[idx]["train"], Kfold_index_dict[idx]["val"], Kfold_index_dict[idx]["test"]
        
        np.random.shuffle(train_index)
        np.random.shuffle(val_index)
        np.random.shuffle(test_index)                        
        
        ################################
        # Define Binding Affinity model
        ################################
        BAModel = PretrainedBA(config, device).cuda()
        checkpoint = torch.load(config["Path"]["pretrained_interaction_predictor"])
        state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('ba_predictor')}
        BAModel.load_state_dict(state_dict, strict=False)
        
        #BAModel.compound_encoder.eval()
        #BAModel.protein_encoder.eval()
        
        ######################################
        # Parms relatively small learning rate
        ######################################
        parameters1 = [v for k, v in BAModel.named_parameters() if "compound_encoder" in k or "protein_encoder" in k or "cross_encoder" in k or "intersites_predictor.pairwise_compound" in k or "intersites_predictor.pairwise_protein" in k]
        parameters2 = [v for k, v in BAModel.named_parameters() if "ba_predictor" in k or "intersites_predictor.latent_compound" in k or "intersites_predictor.latent_protein" in k]
        optimizer = optim.Adam([{'params':parameters1, "lr":1e-10}, {'params':parameters2, "lr":1e-10}], amsgrad=False)

        ######################
        # Parms for training 
        #######################       
        scheduler_dta = CosineAnnealingWarmUpRestarts(optimizer, T_0=15, T_mult=1, eta_maxes=[5e-4, 1e-3], T_up=1, gamma=0.96) 
        print('model trainable params: ', sum(p.numel() for p in BAModel.parameters() if p.requires_grad))
        print('model Total params: ', sum(p.numel() for p in BAModel.parameters()))
        
        ###################
        # Define dataloader 
        ###################
        DTATrainLoader = DataLoader(Subset(DTADataset, train_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        DTAValLoader = DataLoader(Subset(DTADataset, val_index), batch_size=config['Train']['batch_size'], shuffle=True, collate_fn=pad_data)
        print(f"> Training: {len(train_index)}, Validation: {len(val_index)}, Test: {len(test_index)}")

        trainer = BATrainer(config, BAModel, optimizer, device)

        CASAR2016_DTADataset = TrainDataset(interaction_IDs = CASF2016_Interactions_IDs, labels = CASF2016_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2016_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/CASF2016_pocket.pkl",
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2016_indivisual_nonh_interaction.pkl")
        CASAR2016_Loader = DataLoader(CASAR2016_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        CASR2013_DTADataset = TrainDataset(interaction_IDs = CASF2013_Interactions_IDs, labels = CASF2013_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2013_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/CASF2013_pocket.pkl",   
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2013_indivisual_nonh_interaction.pkl")
        CASR2013_Loader = DataLoader(CASR2013_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        CSAR2014_DTADataset = TrainDataset(interaction_IDs = CSAR2014_Interactions_IDs, labels = CSAR2014_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2014_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/CSAR2014_pocket.pkl",  
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2014_indivisual_nonh_interaction.pkl")
        CSAR2014_Loader = DataLoader(CSAR2014_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        CSAR2012_DTADataset = TrainDataset(interaction_IDs = CSAR2012_Interactions_IDs, labels = CSAR2012_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2012_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/CSAR2012_pocket.pkl",  
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2012_indivisual_nonh_interaction.pkl")
        CSAR2012_Loader = DataLoader(CSAR2012_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        CSARset1_DTADataset = TrainDataset(interaction_IDs = CSARset1_Interactions_IDs, labels = CSARset1_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset1_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/CSARset1_pocket.pkl",      
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset1_indivisual_nonh_interaction.pkl")
        CSARset1_Loader = DataLoader(CSARset1_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        CSARset2_DTADataset = TrainDataset(interaction_IDs = CSARset2_Interactions_IDs, labels = CSARset2_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset2_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/CSARset2_pocket.pkl",  
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset2_indivisual_nonh_interaction.pkl")
        CSARset2_Loader = DataLoader(CSARset2_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        Astex_DTADataset = TrainDataset(interaction_IDs = Astex_Interactions_IDs, labels = Astex_BA_labels, device = device,
                                        protein_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/Astex_protein_features.pkl",
                                        pocket_indices_path = "../results/pre-training/protein/CV1/Astex_pocket.pkl",
                                        compound_features_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/graph_data.pt",
                                        interaction_sites_path = "/home/ssm/data/work/Pseq2affinity/data/PDBbind/Astex_indivisual_nonh_interaction.pkl")
        Astex_Loader = DataLoader(Astex_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)

        ##################
        # DTA Train
        ##################
        for epoch in range(1, config["Train"]["epochs"] + 1):
            torch.manual_seed(epoch + 1)
            
            TrainLoss = trainer.DTATrain(DTATrainLoader)
            print(f"[Train ({epoch})] ToTal loss: {TrainLoss['TotalLoss']:.4f}, BA MSE: {TrainLoss['MSE']:.4f}, BA loss: {TrainLoss['DTALoss']:.4f}, PairCE loss: {TrainLoss['InterSitesLoss']:.4f}, PCC: {TrainLoss['PCC']}")
        
            ValLoss, patience = trainer.DTAEval(DTAValLoader, idx)
            print(f"[Val ({epoch})] ToTal loss: {ValLoss['TotalLoss']:.4f}, BA MSE: {ValLoss['MSE']:.4f}, BA loss: {ValLoss['DTALoss']:.4f}, PairCE loss: {ValLoss['InterSitesLoss']:.4f}, PCC: {ValLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(CASAR2016_Loader)
            print(f"[CASF2016 ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(CASR2013_Loader)
            print(f"[CASF2013 ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(CSAR2014_Loader)
            print(f"[CSAR2014 ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(CSAR2012_Loader)
            print(f"[CSAR2012 ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(CSARset1_Loader)
            print(f"[CSARset1 ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(CSARset2_Loader)
            print(f"[CSARset2 ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")

            TestLoss, pre, la = trainer.DTATest(Astex_Loader)
            print(f"[Astex ({epoch})] BA MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")


            if patience > config["Train"]["patience"]:
                print(f"Validation loss do not improves, stop training")
                break
                
            if scheduler_dta is not None:
                scheduler_dta.step()                 
            print()
      
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_maxes=[0.1, 0.1], T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_maxes = eta_maxes
        self.eta_maxes = eta_maxes
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(eta_max - base_lr) * self.T_cur / self.T_up + base_lr 
                    for eta_max, base_lr in zip(self.eta_maxes, self.base_lrs)]
        else:
            return [base_lr + (eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for eta_max, base_lr in zip(self.eta_maxes, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_maxes = [base_eta_max * (self.gamma ** self.cycle) for base_eta_max in self.base_eta_maxes]
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

if __name__ == "__main__":
    main()