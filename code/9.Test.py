import os
import sys
import yaml
import pickle
import random
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from common.utils import load_cfg
from interaction_modules.loaders import TrainDataset, pad_data
from interaction_modules.models import PretrainedBA
from interaction_modules.trainers import BATrainer
from torch.optim.lr_scheduler import _LRScheduler

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

    #################
    # Define argparse
    ##################
    parser = argparse.ArgumentParser(
            description = "Pseq2Sites predicts binding site based on protein sequence information"
    )
    
    parser.add_argument("--config", "-c", required = True, type = input_check, 
                    help = "The file contains information on the protein sequences to predict binding sites. \
                    (refer to the examples.inp file for a more detailed file format)")

    parser.add_argument("--labels", "-l", required = True, type = bool,
                        help = "labels is True: Binding site information is use for evaluation performance; \
                                labels is False: When protein' binding site information is unknwon; \
                                e.g., -t True" 
                )
    
    ##############
    # Load config
    ##############
    conf_path = ""
    config = load_cfg(conf_path)
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu") 
    
    ############
    # Load data
    ############
    data_df = pd.read_csv("", sep = "\t")
    print(f"Number of complexes: {len(data_df)}")
    PDB_IDs, Uniprot_IDs, Lig_codes, BA_labels = data_df.iloc[:, 0].values, data_df.iloc[:, 1].values, data_df.iloc[:, 2].values, data_df.iloc[:, 3].values
    Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(PDB_IDs, Uniprot_IDs, Lig_codes)]) 
    
    BADataset = TrainDataset(interaction_IDs = Interactions_IDs, labels = BA_labels, device = device,
                                    protein_features_path = "",
                                    pocket_indices_path = "",
                                    compound_features_path = "",
                                    interaction_sites_path = "")
                                    
    ################################
    # Define Binding Affinity model
    ################################
    BAModel = PretrainedBA(config, device).cuda()
    checkpoint = torch.load(f"")
    BAModel.load_state_dict(checkpoint)
    
    for parameter in BAModel.parameters():
        parameter.requires_grad = False
    BAModel.eval()
    
    trainer = BATrainer(config, BAModel, None, device)
    
    ##########
    # BA Test
    ##########
    BALoader = DataLoader(BADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
    TestLoss, predictions, labels = trainer.DTATest(Astex_Loader)
    print(f"[Dataset] MSE: {TestLoss['MSE']:.4f}, PCC: {TestLoss['PCC']}")
    
    ##############
    # Save Results
    ##############
    fwrite("", Interactions_IDs, predictions, labels)
    
def fwrite(fw, interaction_ids, predictions, labels):
    fw.write(f"PDB_IDs\tLigand_Codes\tPredictions\tLabels\n")
    
    for ids, prediction, label in zip(interaction_ids, predictions, labels):
        fw.write(f"{ids.split('_')[0]}\t{ids.split('_')[2]}\t{prediction:.4f}\t{label:.4f}\n")
    
    fw.close()

    
if __name__ == "__main__":
    main()