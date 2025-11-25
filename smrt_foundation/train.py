import torch
import argparse
import yaml
import os
import polars as pl
import copy
from enum import StrEnum, auto
from torch import nn
from torch.utils.data import DataLoader
# from torch.amp import GradScaler, autocast
from tqdm import tqdm
from .model import MODEL_REGISTRY
from .loss import cpc_loss
from .dataset import SMRTSequenceDataset, cpc_collate_fn
from functools import partial


def get_args():
    parser = argparse.ArgumentParser(description="Train a CPC model.")
    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--train-data-path', type=str, required=True)
    parser.add_argument('--val-data-path', type=str, required=True)
    parser.add_argument('--stats-path', type=str, required=True)
    parser.add_argument('--output-artifact-path', type=str, required=True)
    parser.add_argument('--output-log-path', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    return parser.parse_args()

def make_dataloader(config, data_path, args, device):
    dataset_params = config['data']
    training_params = config['training']
    
    dataset = SMRTSequenceDataset(
        data_path,
        columns=['seq', 'fi', 'fp', 'ri', 'rp']
    )

    collate_fn = partial(cpc_collate_fn, pad_idx=dataset_params['pad_idx'])
    
    pin_memory = device == 'gpu'
    
    dataloader = DataLoader(
        dataset,
        batch_size=training_params['batch_size'],
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn
    )

    return dataloader


def train_one_epoch(model, dataloader, optimizer, device):
    """executes training on one epoch of training data"""
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="[Train]")
    for batch in pbar:
        seqs = batch['seq_ids'].to(device)
        kins = batch['kinetics'].to(device)
        
        optimizer.zero_grad()
        
        c, predictions = model(seqs, kins)
        c_target = c.detach()
        
        loss = cpc_loss(c_target, predictions)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * seqs.size(0)
        total_samples += seqs.size(0)
        pbar.set_postfix({'train_loss': running_loss / total_samples})

    return running_loss / total_samples

def validate_one_epoch(model, dataloader, device):
    """calculates the mean validation loss on one epoch of data"""
    model.eval()
    val_loss = 0.0
    val_samples = 0
    
    with torch.no_grad():
        pbar_val = tqdm(dataloader, desc="[Val]")
        for batch in pbar_val:
            seqs = batch['seq_ids'].to(device)
            kins = batch['kinetics'].to(device)
            
            c, predictions = model(seqs, kins)
            loss = cpc_loss(c, predictions) 
            
            val_loss += loss.item() * seqs.size(0)
            val_samples += seqs.size(0)
            pbar_val.set_postfix({'val_loss': val_loss / val_samples})

    return val_loss / val_samples


def train(config, device, model, optimizer, train_dl, val_dl):
    epoch_train_losses = []
    epoch_val_losses = []
    epochs = config['training']['epochs']

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        avg_train_loss = train_one_epoch(model, train_dl, optimizer, device)
        epoch_train_losses.append(avg_train_loss)
        
        avg_val_loss = validate_one_epoch(model, val_dl, device)
        epoch_val_losses.append(avg_val_loss)

        print(f'Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}\n')

    print(f'Completed training for {epochs} epochs')
    return {'train_losses': epoch_train_losses, 'val_losses': epoch_val_losses}


def main():
    args = get_args()
    config = parse_yaml(args.config_path)
    config_to_save = copy.deepcopy(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dl = make_dataloader(config, args.train_data_path, args, device)
    val_dl = make_dataloader(config, args.val_data_path, args, device) # <-- Changed
    print("created dataloaders")

    ModelClass = MODEL_REGISTRY[config['model']['architecture']]
    print("instantiated model class")
    model_params = config['model'].get('params', {})
    
    model = ModelClass(**model_params)
    model.to(device)
    
    optimizer = make_optimizer(config['training']['optimizer'], model)

    train_stats = train(config, 
                           device,
                           model, 
                           optimizer, 
                           train_dl, 
                           val_dl 
                           )
                           
    stats_df = pl.DataFrame({
        'epoch': range(1, config['training']['epochs'] + 1),
        'train_loss': train_stats['train_losses'],
        'val_loss': train_stats['val_losses'] 
    })
    
    os.makedirs(os.path.dirname(args.output_artifact_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_log_path), exist_ok=True)

    print(f"Saving training log to {args.output_log_path}")
    stats_df.write_csv(args.output_log_path)

    torch.save({
        'config': config_to_save,
        'model_state_dict': model.state_dict(),
    }, args.output_artifact_path)
    print("Model saved successfully.")
