"""
Training loop implementation.
"""

import os
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """
    General purpose trainer for graph generation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any]
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.current_epoch = 0
        self.global_step = 0
        
        # Setup tensorboard
        log_dir = config.get('log_dir', 'runs')
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Setup checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Customize this based on your model's forward method
            if isinstance(batch, (list, tuple)):
                outputs = self.model(batch[0])
                loss = self.model.compute_loss(outputs, batch[0])
            else:
                outputs = self.model(batch)
                loss = self.model.compute_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (optional)
            if self.config.get('clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['clip_grad_norm']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to tensorboard
            if self.global_step % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(batch, (list, tuple)):
                    outputs = self.model(batch[0])
                    loss = self.model.compute_loss(outputs, batch[0])
                else:
                    outputs = self.model(batch)
                    loss = self.model.compute_loss(outputs, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> None:
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train (uses config if not provided)
        """
        if num_epochs is None:
            num_epochs = self.config.get('num_epochs', 100)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch + 1)
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch + 1)
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch + 1)
                print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
            
            # Save periodic checkpoints
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        self.writer.close()
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.
        
        Args:
            filename: Name of the checkpoint file
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded: {filepath}")
