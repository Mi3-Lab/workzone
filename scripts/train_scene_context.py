"""
Training script for Phase 1.4 Scene Context Classifier.

Quick training on small subset (~5k images per context) using transfer learning.
Expects dataset structure:
  data/04_derivatives/scene_context_dataset/
    highway/
    urban/
    suburban/
    parking/
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import (
    mobilenet_v2,
    resnet18,
    MobileNet_V2_Weights,
    ResNet18_Weights,
)
import wandb

from workzone.models.scene_context import SceneContextConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(backbone: str, num_classes: int, freeze_backbone: bool):
    """Create backbone and replace head for num_classes."""
    backbone = backbone.lower()
    if backbone == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)
        head_params = model.classifier.parameters()
    elif backbone == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)
        head_params = model.fc.parameters()
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze head
        for p in head_params:
            p.requires_grad = True
    return model


def train_scene_context(
    dataset_dir: str = "data/04_derivatives/scene_context_dataset_balanced",
    output_path: str = "weights/scene_context_classifier.pt",
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    use_wandb: bool = False,
    backbone: str = "mobilenet_v2",
    freeze_backbone: bool = False,
    num_workers: int = 4,
    auto_class_weights: bool = False,
):
    """
    Train scene context classifier with transfer learning.
    
    Args:
        dataset_dir: Root directory with subdirectories for each context
        output_path: Where to save trained weights
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: "cuda" or "cpu"
        use_wandb: Log to Weights & Biases
    """
    
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    # Load full dataset - handle incomplete class directories
    try:
        full_dataset = datasets.ImageFolder(
            str(dataset_dir),
            transform=train_transform,
        )
    except FileNotFoundError as e:
        # If some classes are missing, try with only available classes
        print(f"[!] Some classes missing: {e}")
        print("[*] Using available classes only...")
        
        # Find which classes have data
        available_classes = []
        for class_dir in sorted(dataset_dir.iterdir()):
            if class_dir.is_dir() and len(list(class_dir.glob("*"))) > 0:
                available_classes.append(class_dir.name)
        
        if not available_classes:
            raise ValueError(f"No images found in {dataset_dir}")
        
        print(f"[*] Available classes: {available_classes}")
        
        # Create a custom ImageFolder that only loads available classes
        full_dataset = datasets.ImageFolder(
            str(dataset_dir),
            transform=train_transform,
            is_valid_file=lambda x: any(str(x).endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'])
        )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Override val transforms
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Auto-detect num_classes from dataset
    num_classes = len(full_dataset.classes)
    logger.info(f"Detected {num_classes} classes: {full_dataset.classes}")
    
    # Model
    model = build_model(backbone=backbone, num_classes=num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Compute optional class weights for imbalance
    weight_tensor = None
    if auto_class_weights:
        # full_dataset.targets contains class indices
        import numpy as np
        counts = np.bincount(full_dataset.targets, minlength=num_classes)
        # Inverse frequency; normalize to mean=1
        inv = 1.0 / np.clip(counts, 1, None)
        inv = inv * (len(inv) / inv.sum())
        weight_tensor = torch.tensor(inv, dtype=torch.float32, device=device)
        logger.info(f"Using class weights (auto): {inv.tolist()}")

    # Optimizer & loss (trainable params only, handles frozen backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Logging
    if use_wandb:
        wandb.init(
            project="workzone-phase1.4",
            name="scene_context_classifier",
            config={
                "dataset_size": len(full_dataset),
                "train_size": train_size,
                "val_size": val_size,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "backbone": backbone,
                "freeze_backbone": freeze_backbone,
            },
        )
    
    logger.info(f"Training scene context classifier")
    logger.info(f"  Dataset: {len(full_dataset)} images ({train_size} train, {val_size} val)")
    logger.info(f"  Contexts: {SceneContextConfig.CONTEXTS}")
    logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits = model(images)
                loss = loss_fn(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1%} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1%}"
        )
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            logger.info(f"  ✓ Saved best model (val_acc: {val_acc:.1%})")
    
    logger.info(f"✅ Training complete! Best val accuracy: {best_val_acc:.1%}")
    logger.info(f"   Weights saved to: {output_path}")
    
    if use_wandb:
        wandb.finish()
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 1.4 Scene Context Classifier")
    parser.add_argument("--dataset-dir", default="data/04_derivatives/scene_context_dataset_balanced", type=str,
                        help="Path to dataset root with subfolders per class")
    parser.add_argument("--output-path", default="weights/scene_context_classifier.pt", type=str,
                        help="Where to save trained weights")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    parser.add_argument("--backbone", default="mobilenet_v2", choices=["mobilenet_v2", "resnet18"],
                        help="Backbone architecture (pretrained ImageNet)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train head only")
    parser.add_argument("--num-workers", default=4, type=int, help="DataLoader workers")
    parser.add_argument("--auto-class-weights", action="store_true", help="Use inverse-frequency class weights")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"❌ Dataset not found at {dataset_dir}")
        print("   First run: python -c 'from workzone.models.scene_context import create_training_dataset; create_training_dataset(...)'")
        raise SystemExit(1)

    train_scene_context(
        dataset_dir=str(dataset_dir),
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        use_wandb=args.use_wandb,
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
        num_workers=args.num_workers,
        auto_class_weights=args.auto_class_weights,
    )
