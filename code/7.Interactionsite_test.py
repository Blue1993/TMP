import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
import numpy as np
import pandas as pd
#from atom_renamer import AtomicNamer
from rdkit import Chem
import pickle
import os

import os
import sys
import yaml
import pickle
import random
import logging
import numpy as np
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from common.utils import load_cfg

from interaction_modules.loaders import TrainDataset, pad_data
from interaction_modules.models import PretrainedBA
from interaction_modules.trainers import PCITrainer
import pandas as pd
#from torch.optim.lr_scheduler import _LRScheduler
import math

def main():
    
    ##############
    # 1. Load data
    ##############
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

    COACH420_df = pd.read_csv("/home/ssm/data/work/BIBM/data/binding_sites/COACH420_IS_data.tsv", sep = "\t")
    print(f"[COACH420] number of complexes: {len(COACH420_df)}")
    COACH420_PDB_IDs, COACH420_Uniprot_IDs, COACH420_Lig_codes = COACH420_df.iloc[:, 0].values, COACH420_df.iloc[:, 1].values, COACH420_df.iloc[:, 2].values
    COACH420_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(COACH420_PDB_IDs, COACH420_Uniprot_IDs, COACH420_Lig_codes)])
    COACH420_BA_labels = [-1 for i in range(len(COACH420_Interactions_IDs))]
    
    HOLO4K_df = pd.read_csv("/home/ssm/data/work/BIBM/data/binding_sites/HOLO4K_IS_data.tsv", sep = "\t")
    print(f"[HOLO4K] number of complexes: {len(HOLO4K_df)}")
    HOLO4K_PDB_IDs, HOLO4K_Uniprot_IDs, HOLO4K_Lig_codes = HOLO4K_df.iloc[:, 0].values, HOLO4K_df.iloc[:, 1].values, HOLO4K_df.iloc[:, 2].values
    HOLO4K_Interactions_IDs = np.array([f"{p}_{u}_{l}" for p, u, l in zip(HOLO4K_PDB_IDs, HOLO4K_Uniprot_IDs, HOLO4K_Lig_codes)])
    HOLO4K_BA_labels = [-1 for i in range(len(HOLO4K_Interactions_IDs))]

    conf_path = "Interaction_sites_predictor_configuration.yml"
    config = load_cfg(conf_path)    
    device = torch.device("cuda:" + str(config['Train']['device'])) if torch.cuda.is_available() else torch.device("cpu")
    '''
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


    COACH420_DTADataset = TrainDataset(interaction_IDs = COACH420_Interactions_IDs, labels = COACH420_BA_labels, device = device,
                                    protein_features_path = "/home/ssm/data/work/BIBM/data/binding_sites/COACH420_protein_features.pkl",
                                    pocket_indices_path = "../results/pre-training/protein/CV1/COACH420_pocket.pkl",
                                    compound_features_path = "/home/ssm/data/work/BIBM/data/binding_sites/graph_data.pt",
                                    interaction_sites_path = "/home/ssm/data/work/BIBM/data/binding_sites/COACH420_indivisual_nonh_interaction.pkl")
    COACH420_Loader = DataLoader(COACH420_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)


    HOLO4K_DTADataset = TrainDataset(interaction_IDs = HOLO4K_Interactions_IDs, labels = HOLO4K_BA_labels, device = device,
                                    protein_features_path = "/home/ssm/data/work/BIBM/data/binding_sites/HOLO4K_protein_features.pkl",
                                    pocket_indices_path = "../results/pre-training/protein/CV1/HOLO4K_pocket.pkl",
                                    compound_features_path = "/home/ssm/data/work/BIBM/data/binding_sites/graph_data.pt",
                                    interaction_sites_path = "/home/ssm/data/work/BIBM/data/binding_sites/HOLO4K_indivisual_nonh_interaction.pkl")
    HOLO4K_Loader = DataLoader(HOLO4K_DTADataset, batch_size=config['Train']['batch_size'], shuffle=False, collate_fn=pad_data)
    
    ####################
    # 2. Get predictions
    ####################
    for idx in range(0, 1):
        BAAModel = PretrainedBA(config, device).cuda()
        #checkpoint = torch.load(f"/home/ssm/data/work/BIBM/saved_model/interaction/plip/CV{idx}/Pseq2affinity.pth")
        checkpoint = torch.load(f"../checkpoints/pre-training/interaction/CV{idx}/InteractionSite_predictor.pth")
        BAAModel.load_state_dict(checkpoint)

        for parameter in BAAModel.parameters():
            parameter.requires_grad = False
        BAAModel.eval()
        
        trainer = PCITrainer(config, BAAModel, None, device)
        
        CASF2016_pairwise_pre, CASF2016_pairwise_mask, CASF2016_pairwise_labels, CASF2016_lengths = trainer.PCITest(CASAR2016_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/CASF2016_attn_results.pkl", "wb") as f:
            pickle.dump((CASF2016_Interactions_IDs, CASF2016_pairwise_pre, CASF2016_lengths), f)

        CASR2013_pairwise_pre, CASR2013_pairwise_mask, CSAR2013_pairwise_labels, CASF2013_lengths = trainer.PCITest(CASR2013_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/CASF2013_attn_results.pkl", "wb") as f:
            pickle.dump((CASF2013_Interactions_IDs, CASR2013_pairwise_pre, CASF2013_lengths), f)
        
        CSAR2014_pairwise_pre, CSAR2014_pairwise_mask, CSAR2014_pairwise_labels, CSAR2014_lengths = trainer.PCITest(CSAR2014_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/CSAR2014_attn_results.pkl", "wb") as f:
            pickle.dump((CSAR2014_Interactions_IDs, CSAR2014_pairwise_pre, CSAR2014_lengths), f)
        
        CSAR2012_pairwise_pre, CSAR2012_pairwise_mask, CSAR2012_pairwise_labels, CSAR2012_lengths = trainer.PCITest(CSAR2012_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/CSAR2012_attn_results.pkl", "wb") as f:
            pickle.dump((CSAR2012_Interactions_IDs, CSAR2012_pairwise_pre, CSAR2012_lengths), f)
        
        CSARset1_pairwise_pre, CSARset1_pairwise_mask, CSARset1_pairwise_labels, CSARset1_lengths = trainer.PCITest(CSARset1_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/CSARset1_attn_results.pkl", "wb") as f:
            pickle.dump((CSARset1_Interactions_IDs, CSARset1_pairwise_pre, CSARset1_lengths), f)
        
        CSARset2_pairwise_pre, CSARset2_pairwise_mask, CSARset2_pairwise_labels, CSARset2_lengths = trainer.PCITest(CSARset2_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/CSARset2_attn_results.pkl", "wb") as f:
            pickle.dump((CSARset2_Interactions_IDs, CSARset2_pairwise_pre, CSARset2_lengths), f)
        
        Astex_pairwise_pre, Astex_pairwise_mask, Astex_pairwise_labels, Astex_lengths = trainer.PCITest(Astex_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/Astex_attn_results.pkl", "wb") as f:
            pickle.dump((Astex_Interactions_IDs, Astex_pairwise_pre, Astex_lengths), f)
            
        COACH420_pairwise_pre, COACH420_pairwise_mask, COACH420_pairwise_labels, COACH420_lengths = trainer.PCITest(COACH420_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/COACH420_attn_results.pkl", "wb") as f:
            pickle.dump((COACH420_Interactions_IDs, COACH420_pairwise_pre, COACH420_lengths), f)
        
        HOLO4K_pairwise_pre, HOLO4K_pairwise_mask, HOLO4K_pairwise_labels, HOLO4K_lengths = trainer.PCITest(HOLO4K_Loader)
        with open(f"../results/pre-training/interaction/CV{idx}/HOLO4K_attn_results.pkl", "wb") as f:
            pickle.dump((HOLO4K_Interactions_IDs, HOLO4K_pairwise_pre, HOLO4K_lengths), f)
    '''
    
    #################
    # 3. Get results
    #################
    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    def sorted_pred_labels(pred, labels, selected_count):
        results = [(p, l) for p, l in zip(pred, labels)]
        results.sort(key = lambda a: a[0], reverse = True)
        
        #sorted_pred = [i[0] for i in results]
        sorted_labels = np.array([i[1] for i in results])
        sorted_pred = np.zeros(sorted_labels.shape[0]).astype(np.int32)
        sorted_pred[:selected_count] = 1
        
        return sorted_pred, sorted_labels
        
    def get_pair_metric(pairwise_pred, labels):
        #vertax_mask, seqs_mask, pairwise_pred = pred[0], pred[1], pred[2]
        #num_vertex = int(np.sum(vertax_mask))
        #num_residue = int(np.sum(seqs_mask))

        num_vertex = labels.shape[0]
        num_residue = labels.shape[1]
        
        pairwise_pred = pairwise_pred[:num_vertex, :num_residue].reshape(-1)
        pairwise_label = labels.reshape(-1)

        total_count = pairwise_label.shape[0]
        #pos_ratio, neg_ratio = 1, 774.63
        pos_ratio, neg_ratio = 1, 719.89
        
        select_count = np.floor((total_count * pos_ratio)/(pos_ratio + neg_ratio))

        AUROC = roc_auc_score(pairwise_label, pairwise_pred)

        precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
        AUPRC = auc(recall, precision)

        sorted_preds, sorted_labels = sorted_pred_labels(pairwise_pred, pairwise_label, int(select_count))

        tn, fp, fn, tp = confusion_matrix(sorted_labels, sorted_preds).ravel()
        precision = precision_score(sorted_labels, sorted_preds)
        recall = recall_score(sorted_labels, sorted_preds)
        f1 = f1_score(sorted_labels, sorted_preds)
        f2 = fbeta_score(sorted_labels, sorted_preds, beta = 2.0)
        specificity = tn / (tn+fp)
        gmean = np.sqrt(recall * specificity)
        
        return AUROC, AUPRC, precision/(pos_ratio / (pos_ratio + neg_ratio)), precision, recall, specificity, f1, f2, gmean
        
    def get_protein_metric(pairwise_pred, labels):
        #vertax_mask, seqs_mask, pairwise_pred = pred[0], pred[1], pred[2]
        #num_vertex = int(np.sum(vertax_mask))
        #num_residue = int(np.sum(seqs_mask))

        num_vertex = labels.shape[0]
        num_residue = labels.shape[1]
        
        pairwise_pred = np.max(pairwise_pred[:num_vertex, :num_residue], axis = 0)
        pairwise_label = np.clip(np.sum(labels, axis = 0), 0, 1)

        total_count = pairwise_label.shape[0]
        #pos_ratio, neg_ratio = 1, 43.35
        pos_ratio, neg_ratio = 1, 41.39
        
        select_count = np.floor((total_count * pos_ratio)/(pos_ratio + neg_ratio))

        AUROC = roc_auc_score(pairwise_label, pairwise_pred)

        precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
        AUPRC = auc(recall, precision)
        
        sorted_preds, sorted_labels = sorted_pred_labels(pairwise_pred, pairwise_label, int(select_count))

        tn, fp, fn, tp = confusion_matrix(sorted_labels, sorted_preds).ravel()
        precision = precision_score(sorted_labels, sorted_preds)
        recall = recall_score(sorted_labels, sorted_preds)
        f1 = f1_score(sorted_labels, sorted_preds)
        f2 = fbeta_score(sorted_labels, sorted_preds, beta = 2.0)
        specificity = tn / (tn+fp)
        gmean = np.sqrt(recall * specificity)
        
        return AUROC, AUPRC, precision/(pos_ratio / (pos_ratio + neg_ratio)), precision, recall, specificity, f1, f2, gmean
        
        
    def get_compound_metric(pairwise_pred, labels):
        #vertax_mask, seqs_mask, pairwise_pred = pred[0], pred[1], pred[2]
        #num_vertex = int(np.sum(vertax_mask))
        #num_residue = int(np.sum(seqs_mask))
        
        num_vertex = labels.shape[0]
        num_residue = labels.shape[1]
        
        pairwise_pred = np.max(pairwise_pred[:num_vertex, :num_residue], axis = 1)
        pairwise_label = np.clip(np.sum(labels, axis = 1), 0, 1)
        
        total_count = pairwise_label.shape[0]
        #pos_ratio, neg_ratio = 1, 1.61
        pos_ratio, neg_ratio = 1, 1.50
        
        select_count = np.floor((total_count * pos_ratio)/(pos_ratio + neg_ratio))

        try:
            AUROC = roc_auc_score(pairwise_label, pairwise_pred)

        except:
            return None, None, None, None, None, None, None, None, None

        precision, recall, thresholds = precision_recall_curve(pairwise_label, pairwise_pred)
        AUPRC = auc(recall, precision)

        sorted_preds, sorted_labels = sorted_pred_labels(pairwise_pred, pairwise_label, int(select_count))

        tn, fp, fn, tp = confusion_matrix(sorted_labels, sorted_preds).ravel()
        precision = precision_score(sorted_labels, sorted_preds)
        recall = recall_score(sorted_labels, sorted_preds)
        f1 = f1_score(sorted_labels, sorted_preds)
        f2 = fbeta_score(sorted_labels, sorted_preds, beta = 2.0)
        specificity = tn / (tn+fp)
        gmean = np.sqrt(recall * specificity)

        return AUROC, AUPRC, precision/(pos_ratio / (pos_ratio + neg_ratio)), precision, recall, specificity, f1, f2, gmean
        
    def get_results(method, interaction_ids, interaction_site_labels_path, interaction_site_predictions_path):
        
        data = {"PDBID":[], "LigID":[], "AUROC":[], "AUPRC":[], "Precision_Enrichment":[], 
                "Precision":[], "Recall":[], "Specificity":[],
                "F1":[], "F2":[], "GMEAN":[]}
         
        with open(f"{interaction_site_predictions_path}", "rb") as f:
            interaction_site_predictions = pickle.load(f)
        
        with open(f"{interaction_site_labels_path}", "rb") as f:
            interaction_site_labels = pickle.load(f)
        
        #interaction_ids = [f"{i.split('_')[0]}_{i.split('_')[2]}" for i in interaction_ids]
        
        for idx, sample_key in enumerate(interaction_site_predictions[0]):
        #for sample_key in list(interaction_site_labels.keys()):
            #if sample_key in interaction_site_predictions[idx]:
            #if sample_key in interaction_site_predictions:
            sample_key = f'{sample_key.split("_")[0]}_{sample_key.split("_")[2]}'
            #sample_pred, sample_labels = interaction_site_predictions[idx][sample_key], interaction_site_labels[sample_key][1]
            sample_pred, sample_labels = interaction_site_predictions[1][idx], interaction_site_labels[sample_key][1]

            if method == "compound":
                AUROC, AUPRC, precision_Enrichment, precision, recall, specificity, F1, F2, Gmean = get_compound_metric(sample_pred, sample_labels)

            elif method == "protein":
                AUROC, AUPRC, precision_Enrichment, precision, recall, specificity, F1, F2, Gmean = get_protein_metric(sample_pred, sample_labels)

            elif method == "pair":
                AUROC, AUPRC, precision_Enrichment, precision, recall, specificity, F1, F2, Gmean = get_pair_metric(sample_pred, sample_labels)

            #else:
            #    AUROC, AUPRC, precision_Enrichment, precision, recall, specificity, F1, F2, Gmean = 0, 0, 0, 0, 0, 0, 0, 0, 0

            data["PDBID"].append(sample_key.split("_")[0])
            data["LigID"].append(sample_key.split("_")[1])
            data["AUROC"].append(AUROC)
            data["AUPRC"].append(AUPRC)
            data["Precision_Enrichment"].append(precision_Enrichment)
            data["Precision"].append(precision)
            data["Recall"].append(recall)
            data["Specificity"].append(specificity)
            data["F1"].append(F1)
            data["F2"].append(F2)
            data["GMEAN"].append(Gmean)

        return pd.DataFrame(data)
        
    # 3.1 Compound
    for idx in range(0, 1):
        print(f"Start compound results CV{idx}")
        #src_path = "/data2/Attention_results/Pseq2affinity/"
        #save_path = "/home/ssm/data/work/work/Pseq2affinity/NMI_results/results/InteractionSites/BlendNet/"
        
        # CASF2016
        compound_results_df = get_results("compound", CASF2016_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2016_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CASF2016_attn_results.pkl")
        print(f"[CASF2016] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # CASF2013
        compound_results_df = get_results("compound", CASF2013_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2013_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CASF2013_attn_results.pkl")
        print(f"[CASF2013] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # CSAR2014
        compound_results_df = get_results("compound", CSAR2014_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2014_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSAR2014_attn_results.pkl")
        print(f"[CSAR2014] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # CSAR2012
        compound_results_df = get_results("compound", CSAR2012_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2012_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSAR2012_attn_results.pkl")
        print(f"[CSAR2012] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # CSARset1
        compound_results_df = get_results("compound", CSARset1_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset1_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSARset1_attn_results.pkl")
        print(f"[CSARset1] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # CSARset2
        compound_results_df = get_results("compound", CSARset2_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset2_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSARset2_attn_results.pkl")
        print(f"[CSARset2] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # Astex
        compound_results_df = get_results("compound", Astex_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/Astex_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/Astex_attn_results.pkl")
        print(f"[Astex] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # COACH420
        compound_results_df = get_results("compound", COACH420_Interactions_IDs, "/home/ssm/data/work/BIBM/data/binding_sites/COACH420_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/COACH420_attn_results.pkl")
        print(f"[COACH420] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")

        # HOLO4K
        compound_results_df = get_results("compound", HOLO4K_Interactions_IDs, "/home/ssm/data/work/BIBM/data/binding_sites/HOLO4K_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/HOLO4K_attn_results.pkl")
        print(f"[HOLO4K] AUROC: {compound_results_df['AUROC'].mean():.4f}, AUPRC: {compound_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {compound_results_df['Precision_Enrichment'].mean():.4f}, Recall: {compound_results_df['Recall'].mean():.4f}, Precision: {compound_results_df['Precision'].mean():.4f}, F1: {compound_results_df['F1'].mean():.4f}, F2: {compound_results_df['F2'].mean():.4f}, G-Mean: {compound_results_df['GMEAN'].mean():.4f}")
        print()

    print()
    # 3.2 Protein
    for idx in range(0, 1):
        print(f"Start protein CV{idx}")
        #src_path = "/data2/Attention_results/Pseq2affinity/"
        #save_path = "/home/ssm/data/work/work/Pseq2affinity/NMI_results/results/InteractionSites/BlendNet/"
        
        # CASF2016
        protein_results_df = get_results("protein", CASF2016_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2016_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/CASF2016_attn_results.pkl")
        print(f"[CASF2016] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # CASF2013
        protein_results_df = get_results("protein", CASF2013_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2013_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/CASF2013_attn_results.pkl")
        print(f"[CASF2013] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # CSAR2014
        protein_results_df = get_results("protein", CSAR2014_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2014_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/CSAR2014_attn_results.pkl")
        print(f"[CSAR2014] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # CSAR2012
        protein_results_df = get_results("protein", CSAR2012_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2012_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/CSAR2012_attn_results.pkl")
        print(f"[CSAR2012] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # CSARset1
        protein_results_df = get_results("protein", CSARset1_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset1_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/CSARset1_attn_results.pkl")
        print(f"[CSARset1] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # CSARset2
        protein_results_df = get_results("protein", CSARset2_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset2_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/CSARset2_attn_results.pkl")
        print(f"[CSARset2] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # Astex
        protein_results_df = get_results("protein", Astex_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/Astex_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/Astex_attn_results.pkl")
        print(f"[Astex] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # COACH420
        protein_results_df = get_results("protein", COACH420_Interactions_IDs, "/home/ssm/data/work/BIBM/data/binding_sites/COACH420_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/COACH420_attn_results.pkl")
        print(f"[COACH420] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        
        # HOLO4K
        protein_results_df = get_results("protein", HOLO4K_Interactions_IDs, "/home/ssm/data/work/BIBM/data/binding_sites/HOLO4K_indivisual_nonh_interaction.pkl", 
            f"../results/pre-training/interaction/CV{idx}/HOLO4K_attn_results.pkl")
        print(f"[HOLO4K] AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        print()
        #protein_results_df.to_csv(f"{save_path}/CV{idx}_protein_results.tsv", sep = "\t", index = False)
        #print(f"AUROC: {protein_results_df['AUROC'].mean():.4f}, AUPRC: {protein_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {protein_results_df['Precision_Enrichment'].mean():.4f}, Recall: {protein_results_df['Recall'].mean():.4f}, Precision: {protein_results_df['Precision'].mean():.4f}, F1: {protein_results_df['F1'].mean():.4f}, F2: {protein_results_df['F2'].mean():.4f}, G-Mean: {protein_results_df['GMEAN'].mean():.4f}")
        #print()
        
    print()
    # 3.3 Pair
    for idx in range(0, 1):
        print(f"Start pair CV{idx}")
        #src_path = "/data2/Attention_results/Pseq2affinity/"
        #save_path = "/home/ssm/data/work/work/Pseq2affinity/NMI_results/results/InteractionSites/BlendNet/"
        
        # CASF2016
        pair_results_df = get_results("pair", CASF2016_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2016_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CASF2016_attn_results.pkl")
        print(f"[CASF2016] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        
        # CASF2013
        pair_results_df = get_results("pair", CASF2013_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CASF2013_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CASF2013_attn_results.pkl")
        print(f"[CASF2013] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")

        # CSAR2014
        pair_results_df = get_results("pair", CSAR2014_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2014_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSAR2014_attn_results.pkl")
        print(f"[CSAR2014] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")

        # CSAR2012
        pair_results_df = get_results("pair", CSAR2012_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSAR2012_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSAR2012_attn_results.pkl")
        print(f"[CSAR2012] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        
        # CSARset1
        pair_results_df = get_results("pair", CSARset1_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset1_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSARset1_attn_results.pkl")
        print(f"[CSARset1] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        
        # CSARset2
        pair_results_df = get_results("pair", CSARset2_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/CSARset2_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/CSARset2_attn_results.pkl")
        print(f"[CSARset2] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        
        # Astex
        pair_results_df = get_results("pair", Astex_Interactions_IDs, "/home/ssm/data/work/Pseq2affinity/data/PDBbind/Astex_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/Astex_attn_results.pkl")
        print(f"[Astex] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        
        # COACH420
        pair_results_df = get_results("pair", COACH420_Interactions_IDs, "/home/ssm/data/work/BIBM/data/binding_sites/COACH420_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/COACH420_attn_results.pkl")
        print(f"[COACH420] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        
        # HOLO4K
        pair_results_df = get_results("pair", HOLO4K_Interactions_IDs, "/home/ssm/data/work/BIBM/data/binding_sites/HOLO4K_indivisual_nonh_interaction.pkl", 
                f"../results/pre-training/interaction/CV{idx}/HOLO4K_attn_results.pkl")
        print(f"[HOLO4K] AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        print()
        #pair_results_df.to_csv(f"{save_path}/CV{idx}_pair_results.tsv", sep = "\t", index = False)
        #print(f"AUROC: {pair_results_df['AUROC'].mean():.4f}, AUPRC: {pair_results_df['AUPRC'].mean():.4f}, Precision_Enrichment: {pair_results_df['Precision_Enrichment'].mean():.4f}, Recall: {pair_results_df['Recall'].mean():.4f}, Precision: {pair_results_df['Precision'].mean():.4f}, F1: {pair_results_df['F1'].mean():.4f}, F2: {pair_results_df['F2'].mean():.4f}, G-Mean: {pair_results_df['GMEAN'].mean():.4f}")
        #print()
    
if __name__ == "__main__":
    main()
