#!/usr/bin/env python3
"""
Phase 2.1: Train Temporal Attention Aggregator

Train the attention-weighted temporal model on CSV timelines
from process_video_fusion.py outputs.

Usage:
    python tools/train_temporal_attention.py \\
        --train-dir outputs/phase1_4_demo \\
        --val-dir outputs/phase1_4_eval \\
        --output-dir runs/phase2_1 \\
        --epochs 50 \\
        --batch-size 32
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from workzone.models.temporal_attention import (
    TemporalAttentionAggregator,
    TemporalAttentionConfig,
)
from workzone.data.temporal_dataset import create_dataloaders
from workzone.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class Phase21Trainer:
    """Trainer for Phase 2.1 temporal attention model."""
    
    def __init__(
        self,
        model: TemporalAttentionAggregator,
        train_loader,
        val_loader,
        device: str,
        output_dir: Path,
        learning_rate: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 50,  # 50 epochs
            eta_min=1e-6,
        )
        
        # Loss functions
        self.confidence_criterion = nn.BCELoss()
        self.state_criterion = nn.CrossEntropyLoss()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(output_dir / "logs"))
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_confidence_loss = 0.0
        total_state_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Move to device
            cue_scores = batch['cue_scores'].to(self.device)
            scene_logits = batch['scene_logits'].to(self.device)
            yolo_scores = batch['yolo_scores'].to(self.device)
            clip_scores = batch['clip_scores'].to(self.device)
            motion_plausibility = batch['motion_plausibility'].to(self.device)
            frame_deltas = batch['frame_deltas'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            states = batch['states'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                cue_scores=cue_scores,
                scene_logits=scene_logits,
                yolo_scores=yolo_scores,
                clip_scores=clip_scores,
                motion_plausibility=motion_plausibility,
                frame_deltas=frame_deltas,
                mask=mask,
            )
            
            # Compute losses
            confidence_loss = self.confidence_criterion(outputs['confidence'], labels)
            state_loss = self.state_criterion(outputs['state_logits'], states)
            
            loss = (
                TemporalAttentionConfig.CONFIDENCE_LOSS_WEIGHT * confidence_loss +
                TemporalAttentionConfig.STATE_LOSS_WEIGHT * state_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            total_confidence_loss += confidence_loss.item()
            total_state_loss += state_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'conf_loss': confidence_loss.item(),
                'state_loss': state_loss.item(),
            })
            
            # TensorBoard
            self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            self.writer.add_scalar('train/confidence_loss', confidence_loss.item(), self.global_step)
            self.writer.add_scalar('train/state_loss', state_loss.item(), self.global_step)
            self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            self.global_step += 1
        
        return {
            'loss': total_loss / n_batches,
            'confidence_loss': total_confidence_loss / n_batches,
            'state_loss': total_state_loss / n_batches,
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_confidence_loss = 0.0
        total_state_loss = 0.0
        n_batches = 0
        
        correct_states = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                # Move to device
                cue_scores = batch['cue_scores'].to(self.device)
                scene_logits = batch['scene_logits'].to(self.device)
                yolo_scores = batch['yolo_scores'].to(self.device)
                clip_scores = batch['clip_scores'].to(self.device)
                motion_plausibility = batch['motion_plausibility'].to(self.device)
                frame_deltas = batch['frame_deltas'].to(self.device)
                mask = batch['mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                states = batch['states'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    cue_scores=cue_scores,
                    scene_logits=scene_logits,
                    yolo_scores=yolo_scores,
                    clip_scores=clip_scores,
                    motion_plausibility=motion_plausibility,
                    frame_deltas=frame_deltas,
                    mask=mask,
                )
                
                # Compute losses
                confidence_loss = self.confidence_criterion(outputs['confidence'], labels)
                state_loss = self.state_criterion(outputs['state_logits'], states)
                
                loss = (
                    TemporalAttentionConfig.CONFIDENCE_LOSS_WEIGHT * confidence_loss +
                    TemporalAttentionConfig.STATE_LOSS_WEIGHT * state_loss
                )
                
                # Metrics
                total_loss += loss.item()
                total_confidence_loss += confidence_loss.item()
                total_state_loss += state_loss.item()
                n_batches += 1
                
                # State accuracy
                pred_states = outputs['state_logits'].argmax(dim=-1)
                correct_states += (pred_states == states).sum().item()
                total_samples += states.size(0)
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'state_acc': correct_states / total_samples,
                })
        
        metrics = {
            'loss': total_loss / n_batches,
            'confidence_loss': total_confidence_loss / n_batches,
            'state_loss': total_state_loss / n_batches,
            'state_accuracy': correct_states / total_samples,
        }
        
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
            logger.info(f"✓ Saved best model (val_loss={metrics['loss']:.4f})")
    
    def train(self, num_epochs: int):
        """Full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Conf: {train_metrics['confidence_loss']:.4f}, "
                       f"State: {train_metrics['state_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate(epoch)
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Conf: {val_metrics['confidence_loss']:.4f}, "
                       f"State: {val_metrics['state_loss']:.4f}, "
                       f"Acc: {val_metrics['state_accuracy']:.2%}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Save every 10 epochs
            if epoch % 10 == 0:
                torch.save(
                    {'model_state_dict': self.model.state_dict()},
                    self.output_dir / f'checkpoint_epoch{epoch}.pt'
                )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"{'='*60}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2.1 temporal attention model")
    
    parser.add_argument(
        "--train-dir",
        type=str,
        required=True,
        help="Directory containing training CSV files"
    )
    
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Directory containing validation CSV files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/phase2_1",
        help="Output directory for checkpoints and logs"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=90,
        help="Temporal window size (frames)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device"
    )
    
    args = parser.parse_args()
    
    # Find CSV files
    train_csvs = list(Path(args.train_dir).rglob("*timeline_fusion.csv"))
    val_csvs = list(Path(args.val_dir).rglob("*timeline_fusion.csv"))
    
    logger.info(f"Found {len(train_csvs)} training CSVs")
    logger.info(f"Found {len(val_csvs)} validation CSVs")
    
    if not train_csvs or not val_csvs:
        logger.error("No CSV files found! Please run process_video_fusion.py first.")
        sys.exit(1)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_csvs=train_csvs,
        val_csvs=val_csvs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=4,
    )
    
    # Create model
    model = TemporalAttentionAggregator(
        d_model=TemporalAttentionConfig.D_MODEL,
        nhead=TemporalAttentionConfig.NHEAD,
        num_layers=TemporalAttentionConfig.NUM_LAYERS,
        dim_feedforward=TemporalAttentionConfig.DIM_FEEDFORWARD,
        dropout=TemporalAttentionConfig.DROPOUT,
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Train
    trainer = Phase21Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        output_dir=Path(args.output_dir),
        learning_rate=args.learning_rate,
    )
    
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
