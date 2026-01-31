#!/usr/bin/env python3
"""
Phase 2.1 Training Script using Real Trajectory Data

Trains TemporalAttentionAggregator on actual ROADWork annotated trajectories.
Uses:
- Ground truth object categories (45 types -> 5 cue groups)
- Scene descriptions (text context)
- Temporal sequences (90+ frames from sparse 5-frame trajectories)
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

from workzone.models.temporal_attention import TemporalAttentionAggregator
from workzone.data.trajectory_dataset import create_trajectory_dataloaders

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)


class Phase21TrajectoryTrainer:
    """Train Phase 2.1 model on real trajectory annotations."""
    
    def __init__(
        self,
        model: TemporalAttentionAggregator,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        output_dir: Path = Path('outputs/phase2_1_train'),
        log_interval: int = 50,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        
        self.log_interval = log_interval
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5).to(device))
        self.ce_loss = nn.CrossEntropyLoss()
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=self.output_dir / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_bce_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            cue_scores = batch['cue_scores'].to(self.device)  # [B, T, 5]
            scene_logits = batch['scene_logits'].to(self.device)  # [B, T, 3]
            labels = batch['labels'].to(self.device)  # [B]
            mask = batch['mask'].to(self.device)  # [B, T] bool
            
            batch_size = cue_scores.size(0)
            seq_len = cue_scores.size(1)
            
            # Model forward pass
            # Pad additional features to match expected input
            yolo_conf = torch.ones(batch_size, seq_len, 1).to(self.device)  # Generic YOLO present
            clip_conf = torch.ones(batch_size, seq_len, 1).to(self.device)  # Generic CLIP match
            motion_score = torch.zeros(batch_size, seq_len, 1).to(self.device)  # No motion from static data
            
            # Temporal position encoding
            frame_positions = torch.arange(seq_len, dtype=torch.float32).to(self.device)
            frame_positions = (frame_positions / seq_len).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
            
            # Concatenate all features: [B, T, 5+3+1+1+1+1=12]
            combined_features = torch.cat([
                cue_scores,
                scene_logits,
                yolo_conf,
                clip_conf,
                motion_score,
                frame_positions,
            ], dim=-1)
            
            # Model inference
            confidence_logits, state_logits, attention_weights = self.model(
                combined_features,
                mask=mask
            )
            
            # Loss: confidence prediction
            confidence_pred = confidence_logits.squeeze(-1)  # [B]
            bce_loss = self.bce_loss(confidence_pred, labels)
            
            # Loss: state prediction (optional - use 0/1 label mapped to state)
            state_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            state_labels[labels > 0.5] = 1  # State 1 = inside workzone
            ce_loss = self.ce_loss(state_logits, state_labels)
            
            # Combined loss
            loss = bce_loss + 0.2 * ce_loss  # Weight state loss lower
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item() * batch_size
            total_bce_loss += bce_loss.item() * batch_size
            total_ce_loss += ce_loss.item() * batch_size
            total_samples += batch_size
            
            self.global_step += 1
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / total_samples
                avg_bce = total_bce_loss / total_samples
                avg_ce = total_ce_loss / total_samples
                
                logger.info(
                    f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, CE: {avg_ce:.4f})"
                )
                
                self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                self.writer.add_scalar('train/bce_loss', avg_bce, self.global_step)
                self.writer.add_scalar('train/ce_loss', avg_ce, self.global_step)
        
        avg_loss = total_loss / total_samples
        avg_bce = total_bce_loss / total_samples
        avg_ce = total_ce_loss / total_samples
        
        return {
            'loss': avg_loss,
            'bce_loss': avg_bce,
            'ce_loss': avg_ce,
        }
    
    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                cue_scores = batch['cue_scores'].to(self.device)
                scene_logits = batch['scene_logits'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                batch_size = cue_scores.size(0)
                seq_len = cue_scores.size(1)
                
                # Prepare features
                yolo_conf = torch.ones(batch_size, seq_len, 1).to(self.device)
                clip_conf = torch.ones(batch_size, seq_len, 1).to(self.device)
                motion_score = torch.zeros(batch_size, seq_len, 1).to(self.device)
                
                frame_positions = torch.arange(seq_len, dtype=torch.float32).to(self.device)
                frame_positions = (frame_positions / seq_len).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
                
                combined_features = torch.cat([
                    cue_scores,
                    scene_logits,
                    yolo_conf,
                    clip_conf,
                    motion_score,
                    frame_positions,
                ], dim=-1)
                
                # Forward pass
                confidence_logits, state_logits, _ = self.model(combined_features, mask=mask)
                
                confidence_pred = confidence_logits.squeeze(-1)
                loss = self.bce_loss(confidence_pred, labels)
                
                # Accuracy
                preds = (torch.sigmoid(confidence_pred) > 0.5).float()
                correct = (preds == labels).sum().item()
                
                total_loss += loss.item() * batch_size
                total_correct += correct
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        logger.info(f"[Validation Epoch {epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
        
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/accuracy', accuracy, self.global_step)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
        }
    
    def train(self, train_loader, val_loader, epochs: int = 50):
        """Full training loop."""
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, val_metrics)
                logger.info(f"✓ New best validation loss: {val_metrics['loss']:.4f}")
        
        logger.info("Training complete!")
        self.writer.close()
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # Latest
        latest_path = self.output_dir / 'phase2_1_latest.pt'
        torch.save(checkpoint, latest_path)
        logger.info(f"Saved checkpoint to {latest_path}")
        
        # Best
        best_path = self.output_dir / 'phase2_1_best.pt'
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(
        description='Train Phase 2.1 Temporal Attention on ROADWork Trajectories'
    )
    parser.add_argument(
        '--train-annotation',
        type=Path,
        default=Path('data/01_raw/trajectories/sparse/annotations/trajectories_train_equidistant.json'),
        help='Path to training annotations',
    )
    parser.add_argument(
        '--val-annotation',
        type=Path,
        default=Path('data/01_raw/trajectories/sparse/annotations/trajectories_val_equidistant.json'),
        help='Path to validation annotations',
    )
    parser.add_argument(
        '--image-dir',
        type=Path,
        default=Path('data/01_raw/trajectories/sparse'),
        help='Path to image directory',
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=18,  # 3.6 seconds @ 5 fps (sparse trajectories)
        help='Sequence length (frames)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader workers',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/phase2_1_train'),
        help='Output directory',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)',
    )
    
    args = parser.parse_args()
    
    # Check data exists
    if not args.train_annotation.exists():
        logger.error(f"Train annotation not found: {args.train_annotation}")
        return 1
    
    if not args.val_annotation.exists():
        logger.error(f"Val annotation not found: {args.val_annotation}")
        return 1
    
    logger.info(f"Train annotation: {args.train_annotation}")
    logger.info(f"Val annotation: {args.val_annotation}")
    logger.info(f"Image directory: {args.image_dir}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_trajectory_dataloaders(
        train_annotation=args.train_annotation,
        val_annotation=args.val_annotation,
        image_dir=args.image_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Create model
    logger.info("Creating model...")
    device = torch.device(args.device)
    model = TemporalAttentionAggregator(
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    )
    
    # Create trainer
    trainer = Phase21TrajectoryTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
