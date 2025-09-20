"""
Training Pipeline for CST Models
Implements contrastive pre-training, language modeling, and fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import json
import logging
import wandb
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import time

from cst_transformer import CSTransformer
from config import CSTConfig, TrainingConfig


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, positive_pairs: torch.Tensor, negative_pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positive_pairs: [batch_size, embedding_dim] - Positive examples
            negative_pairs: [batch_size * num_negatives, embedding_dim] - Negative examples
        """
        batch_size = positive_pairs.size(0)
        
        # Normalize embeddings
        positive_pairs = F.normalize(positive_pairs, dim=1)
        negative_pairs = F.normalize(negative_pairs, dim=1)
        
        # Compute similarities
        pos_sim = torch.sum(positive_pairs * positive_pairs, dim=1) / self.temperature  # Self-similarity
        
        # Reshape negatives and compute similarities
        num_negatives = negative_pairs.size(0) // batch_size
        negative_pairs = negative_pairs.view(batch_size, num_negatives, -1)
        
        neg_sims = torch.bmm(
            positive_pairs.unsqueeze(1),
            negative_pairs.transpose(1, 2)
        ).squeeze(1) / self.temperature  # [batch_size, num_negatives]
        
        # Combine positive and negative similarities
        all_sims = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        
        # InfoNCE loss
        labels = torch.zeros(batch_size, dtype=torch.long, device=positive_pairs.device)
        loss = F.cross_entropy(all_sims, labels)
        
        return loss


class SpectralRegularizer:
    """Prevents representation drift and catastrophic forgetting"""
    
    def __init__(self, config):
        self.config = config
        self.reference_embeddings = {}
        self.update_frequency = config.reference_update_freq
        self.step_count = 0
        self.momentum = config.reference_momentum
        
    def compute_drift_loss(self, current_embeddings: torch.Tensor, 
                          fragment_ids: torch.Tensor) -> torch.Tensor:
        """Compute drift regularization loss"""
        drift_loss = torch.tensor(0.0, device=current_embeddings.device, requires_grad=True)
        
        for frag_id in fragment_ids.unique():
            frag_id_item = frag_id.item()
            if frag_id_item in self.reference_embeddings:
                current_mask = fragment_ids == frag_id
                current_repr = current_embeddings[current_mask].mean(0)
                reference_repr = self.reference_embeddings[frag_id_item]
                
                drift_loss = drift_loss + F.mse_loss(current_repr, reference_repr)
                
        return drift_loss / len(fragment_ids.unique()) if len(fragment_ids.unique()) > 0 else drift_loss
    
    def update_references(self, embeddings: torch.Tensor, fragment_ids: torch.Tensor):
        """Update reference embeddings with exponential moving average"""
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            for frag_id in fragment_ids.unique():
                frag_id_item = frag_id.item()
                current_mask = fragment_ids == frag_id
                current_repr = embeddings[current_mask].mean(0).detach()
                
                if frag_id_item in self.reference_embeddings:
                    self.reference_embeddings[frag_id_item] = (
                        self.momentum * self.reference_embeddings[frag_id_item] +
                        (1 - self.momentum) * current_repr
                    )
                else:
                    self.reference_embeddings[frag_id_item] = current_repr


class CSTDataset(Dataset):
    """Dataset class for CST training"""
    
    def __init__(self, data_path: str, config: CSTConfig, split: str = 'train'):
        self.config = config
        self.split = split
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load dataset from file - implement based on your data format"""
        # This is a simplified version - adapt to your data format
        data = []
        
        # Example data loading (replace with actual implementation)
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            # Generate synthetic data for testing
            logger.warning(f"Data file {data_path} not found. Generating synthetic data.")
            data = self._generate_synthetic_data(1000)
        
        return data
    
    def _generate_synthetic_data(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic data for testing"""
        data = []
        for i in range(num_samples):
            seq_len = torch.randint(10, self.config.max_sequence_length, (1,)).item()
            
            sample = {
                'input_ids': torch.randint(1, self.config.vocab_size, (seq_len,)),
                'labels': torch.randint(1, self.config.vocab_size, (seq_len,)),
                'fragment_chars': torch.randint(0, self.config.char_vocab_size, (seq_len, 32)),
                'context_chars': torch.randint(0, self.config.char_vocab_size, (seq_len, 64)),
                'context_data': {
                    'document_embedding': torch.randn(self.config.raw_doc_dim),
                    'metadata': {
                        'author': torch.randint(0, self.config.num_authors, (1,)).item(),
                        'domain': torch.randint(0, self.config.num_domains, (1,)).item(),
                        'timestamp': torch.randn(1).item(),
                    }
                }
            }
            data.append(sample)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batching"""
    batch_size = len(batch)
    
    # Find max sequence length in batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Initialize batch tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    # Fragment and context data
    fragment_chars = torch.zeros(batch_size, max_len, 32, dtype=torch.long)
    context_chars = torch.zeros(batch_size, max_len, 64, dtype=torch.long)
    
    # Context data
    context_data = {
        'document_embedding': torch.zeros(batch_size, batch[0]['context_data']['document_embedding'].size(0)),
        'metadata': {
            'author': torch.zeros(batch_size, dtype=torch.long),
            'domain': torch.zeros(batch_size, dtype=torch.long),
            'timestamp': torch.zeros(batch_size, dtype=torch.float),
        }
    }
    
    # Fill batch
    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        
        input_ids[i, :seq_len] = item['input_ids']
        labels[i, :seq_len] = item['labels']
        attention_mask[i, :seq_len] = 1.0
        
        fragment_chars[i, :seq_len] = item['fragment_chars']
        context_chars[i, :seq_len] = item['context_chars']
        
        context_data['document_embedding'][i] = item['context_data']['document_embedding']
        context_data['metadata']['author'][i] = item['context_data']['metadata']['author']
        context_data['metadata']['domain'][i] = item['context_data']['metadata']['domain']
        context_data['metadata']['timestamp'][i] = item['context_data']['metadata']['timestamp']
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'fragment_chars': fragment_chars,
        'context_chars': context_chars,
        'context_data': context_data
    }


class CSTTrainer:
    """Main trainer class for CST models"""
    
    def __init__(self, 
                 model: CSTransformer, 
                 config: CSTConfig,
                 train_config: TrainingConfig):
        self.model = model
        self.config = config
        self.train_config = train_config
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.contrastive_loss = InfoNCELoss(temperature=config.temperature)
        self.spectral_regularizer = SpectralRegularizer(config)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Setup distributed training if needed
        self.is_distributed = train_config.distributed
        if self.is_distributed:
            self.setup_distributed()
        
        # Setup logging
        if train_config.wandb_project and not self.is_distributed or dist.get_rank() == 0:
            wandb.init(project=train_config.wandb_project, config=config.__dict__)
    
    def setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(backend=self.train_config.backend)
        torch.cuda.set_device(self.train_config.rank)
        self.model = DDP(self.model, device_ids=[self.train_config.rank])
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Separate parameters for different learning rates
        cst_params = []
        transformer_params = []
        
        for name, param in self.model.named_parameters():
            if 'cst_module' in name:
                cst_params.append(param)
            else:
                transformer_params.append(param)
        
        # Different learning rates for CST vs transformer
        param_groups = [
            {'params': cst_params, 'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': transformer_params, 'lr': self.config.learning_rate * 0.5, 'weight_decay': self.config.weight_decay}
        ]
        
        self.optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6)
        
        # Setup scheduler
        total_steps = self.train_config.max_epochs * len(self.train_loader)
        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.config.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - self.config.warmup_steps)
        
        self.scheduler = SequentialLR(
            self.optimizer, 
            [warmup_scheduler, cosine_scheduler], 
            milestones=[self.config.warmup_steps]
        )
    
    def contrastive_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Perform contrastive learning step"""
        input_ids = batch['input_ids']
        context_data = batch['context_data']
        
        # Get CST embeddings for original context
        positive_embeddings = self.model.get_embeddings(input_ids, context_data)
        
        # Create negative contexts (shuffle metadata)
        negative_context_data = context_data.copy()
        batch_size = input_ids.size(0)
        
        # Shuffle authors and domains for negatives
        perm = torch.randperm(batch_size)
        negative_context_data['metadata'] = {
            'author': context_data['metadata']['author'][perm],
            'domain': context_data['metadata']['domain'][perm],
            'timestamp': context_data['metadata']['timestamp'][perm],
        }
        
        # Get embeddings with negative context
        negative_embeddings = self.model.get_embeddings(input_ids, negative_context_data)
        
        # Pool sequence representations
        positive_pooled = positive_embeddings.mean(dim=1)  # [batch_size, d_model]
        negative_pooled = negative_embeddings.mean(dim=1)  # [batch_size, d_model]
        
        # Contrastive loss
        loss = self.contrastive_loss(positive_pooled, negative_pooled)
        
        return loss
    
    def language_modeling_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Perform masked language modeling step"""
        outputs = self.model(
            input_ids=batch['input_ids'],
            context_data=batch['context_data'],
            attention_mask=batch['attention_mask'],
            fragment_chars=batch['fragment_chars'],
            context_chars=batch['context_chars'],
            labels=batch['labels']
        )
        
        mlm_loss = outputs['loss']
        cst_stats = outputs['cst_stats']
        
        # Spectral regularization
        embeddings = outputs['hidden_states']
        drift_loss = self.spectral_regularizer.compute_drift_loss(
            embeddings.view(-1, embeddings.size(-1)),
            batch['input_ids'].view(-1)
        )
        
        # Update reference embeddings
        self.spectral_regularizer.update_references(
            embeddings.view(-1, embeddings.size(-1)),
            batch['input_ids'].view(-1)
        )
        
        total_loss = mlm_loss + self.config.drift_regularization_weight * drift_loss
        
        metrics = {
            'mlm_loss': mlm_loss.item(),
            'drift_loss': drift_loss.item(),
            'cache_hit_rate': cst_stats.get('hit_rate', 0.0),
            'ambiguous_ratio': cst_stats.get('ambiguous_ratio', 0.0)
        }
        
        return total_loss, metrics
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Combined training step with contrastive and MLM losses"""
        self.model.train()
        
        # Contrastive learning
        contrastive_loss = self.contrastive_step(batch)
        
        # Masked language modeling
        mlm_loss, mlm_metrics = self.language_modeling_step(batch)
        
        # Combined loss
        total_loss = (self.config.contrastive_weight * contrastive_loss + 
                     self.config.mlm_weight * mlm_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        
        # Metrics
        metrics = {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            **mlm_metrics
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        val_losses = []
        all_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    context_data=batch['context_data'],
                    attention_mask=batch['attention_mask'],
                    fragment_chars=batch['fragment_chars'],
                    context_chars=batch['context_chars'],
                    labels=batch['labels']
                )
                
                val_losses.append(outputs['loss'].item())
                
                # Collect metrics
                cst_stats = outputs['cst_stats']
                all_metrics['cache_hit_rate'].append(cst_stats.get('hit_rate', 0.0))
                all_metrics['ambiguous_ratio'].append(cst_stats.get('ambiguous_ratio', 0.0))
        
        # Average metrics
        avg_metrics = {
            'val_loss': np.mean(val_losses),
            'val_cache_hit_rate': np.mean(all_metrics['cache_hit_rate']),
            'val_ambiguous_ratio': np.mean(all_metrics['ambiguous_ratio'])
        }
        
        return avg_metrics
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'spectral_regularizer_refs': self.spectral_regularizer.reference_embeddings
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location='cuda')
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.spectral_regularizer.reference_embeddings = checkpoint.get('spectral_regularizer_refs', {})
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Main training loop"""
        self.train_loader = train_loader
        self.setup_optimizer()
        
        logger.info(f"Starting training for {self.train_config.max_epochs} epochs")
        logger.info(f"Total training steps: {len(train_loader) * self.train_config.max_epochs}")
        
        for epoch in range(self.train_config.max_epochs):
            self.epoch = epoch
            
            # Training loop
            self.model.train()
            epoch_metrics = defaultdict(list)
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for batch_idx, batch in enumerate(progress_bar):
                
                # Move batch to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Training step
                step_metrics = self.train_step(batch)
                
                # Update metrics
                for key, value in step_metrics.items():
                    epoch_metrics[key].append(value)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_metrics['total_loss']:.4f}",
                    'lr': f"{step_metrics['learning_rate']:.2e}",
                    'cache_hit': f"{step_metrics['cache_hit_rate']:.2%}"
                })
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.train_config.log_every_n_steps == 0:
                    avg_metrics = {k: np.mean(v[-self.train_config.log_every_n_steps:]) 
                                 for k, v in epoch_metrics.items()}
                    
                    if wandb.run:
                        wandb.log(avg_metrics, step=self.global_step)
                    
                    logger.info(f"Step {self.global_step}: {avg_metrics}")
                
                # Validation
                if (val_loader and 
                    self.global_step % self.train_config.eval_every_n_steps == 0):
                    val_metrics = self.validate(val_loader)
                    
                    if wandb.run:
                        wandb.log(val_metrics, step=self.global_step)
                    
                    logger.info(f"Validation at step {self.global_step}: {val_metrics}")
                    
                    # Save best model
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(
                            f"{self.train_config.checkpoint_dir}/checkpoint_step_{self.global_step}.pt",
                            is_best=True
                        )
                
                # Checkpoint saving
                if self.global_step % self.train_config.save_every_n_steps == 0:
                    self.save_checkpoint(
                        f"{self.train_config.checkpoint_dir}/checkpoint_step_{self.global_step}.pt"
                    )
            
            # End of epoch validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                logger.info(f"End of epoch {epoch} validation: {val_metrics}")
                
                if wandb.run:
                    wandb.log({f"epoch_{k}": v for k, v in val_metrics.items()}, step=self.global_step)


def main():
    """Main training script"""
    # Load configs
    config = CSTConfig()
    train_config = TrainingConfig()
    
    # Setup model
    model = CSTransformer(config, task_type='mlm')
    model.cuda()
    model.enable_cst_profiling(True)
    
    # Setup datasets
    train_dataset = CSTDataset(train_config.train_data_path, config, split='train')
    val_dataset = CSTDataset(train_config.val_data_path, config, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup trainer
    trainer = CSTTrainer(model, config, train_config)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    # Save final model
    model.save_pretrained(f"{train_config.checkpoint_dir}/final_model")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()