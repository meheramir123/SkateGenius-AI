#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import models
from torch.cuda.amp import GradScaler, autocast

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import glob
from tqdm import tqdm
import warnings
import time
import json
from datetime import datetime
warnings.filterwarnings('ignore')


def optimize_gpu():
    """Configure and enable GPU optimizations for maximum training throughput.
    Enables cuDNN benchmarking, sets memory fraction to 85%, activates expandable
    memory segments, and balances CPU thread count. Prints a summary of the
    device configuration on success and returns True if CUDA is available,
    False otherwise.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

        torch.cuda.empty_cache()

        torch.cuda.set_per_process_memory_fraction(0.85)

        torch.cuda.memory._set_allocator_settings('expandable_segments:True')

        torch.set_num_threads(4)

        print("STEADY GPU optimizations enabled (85% TARGET):")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  - cuDNN benchmark: True")
        print(f"  - Memory fraction: 85% (HIGH UTILIZATION TARGET)")
        print(f"  - Memory pool: Enabled")
        print(f"  - Threads: 4 (Balanced CPU/GPU)")
        return True
    return False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


SKATEBOARD_TRICKS = [
    'Ollie', 'Kickflip', 'Shuvit', 'Manual', 'Hardflip',
    '5050grind', '50grind', 'Backside180', 'BacksideAir', 'Boardslide',
    'Boneless180', 'Smithgrind', 'Benihana', 'Impossible', 'Treflip'
]

TRICK_TO_IDX = {trick: idx for idx, trick in enumerate(SKATEBOARD_TRICKS)}
IDX_TO_TRICK = {idx: trick for idx, trick in enumerate(SKATEBOARD_TRICKS)}


FAST_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
FAST_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def fast_frame_transform(np_frame):
    """Convert a raw NumPy video frame to a normalised floating-point tensor.
    Accepts an HxWxC uint8 RGB array, converts it to a CxHxW float32 tensor,
    scales pixel values to [0, 1], then applies ImageNet mean/std normalisation.
    The function is defined at module scope so it remains picklable when used
    inside DataLoader worker processes on Windows.
    """
    t = torch.as_tensor(np_frame, dtype=torch.float32)
    t = t.permute(2, 0, 1)
    t = t / 255.0
    return (t - FAST_IMAGENET_MEAN) / FAST_IMAGENET_STD


class OptimizedVideoDataset(Dataset):
    """PyTorch Dataset that loads fixed-length frame sequences from video files.
    Frames are extracted at evenly spaced temporal positions using OpenCV and
    cached in memory (up to cache_size entries) to avoid redundant disk I/O on
    repeated access. An optional transform callable is applied to every frame
    after extraction. Corrupted or unreadable frames are replaced with the last
    successfully decoded frame, or a black frame when no good frame exists yet.
    """

    def __init__(self, video_paths, labels, sequence_length=16, transform=None, cache_size=100):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        if video_path in self.cache:
            frames = self.cache[video_path]
        else:
            frames = self.extract_frames_optimized(video_path)

            if len(self.cache) < self.cache_size:
                self.cache[video_path] = frames

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        return torch.stack(frames), torch.tensor(label, dtype=torch.long)

    def extract_frames_optimized(self, video_path):
        """Sample exactly sequence_length frames from a video file using OpenCV.
        Frames are chosen at linearly-spaced indices across the full duration.
        Each frame is decoded, colour-converted from BGR to RGB, and resized to
        224×224. Failed reads duplicate the previous valid frame; if no valid
        frame has been decoded yet a black frame is substituted. The returned
        list is guaranteed to contain exactly sequence_length NumPy arrays of
        shape (224, 224, 3) with dtype uint8.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.sequence_length)]

            frame_indices = np.linspace(0, max(total_frames - 1, 0), self.sequence_length, dtype=int)
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
                else:
                    frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

            cap.release()
            if len(frames) < self.sequence_length:
                frames.extend([frames[-1]] * (self.sequence_length - len(frames)))
            elif len(frames) > self.sequence_length:
                frames = frames[:self.sequence_length]
            return frames

        except Exception as e:
            print(f"ERROR extracting frames from {video_path}: {str(e)}")
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.sequence_length)]


def create_optimized_data_loaders(data_dir, batch_size=22, sequence_length=12, train_split=0.8):
    """Build train and validation DataLoaders with stratified class splitting and class-balanced loss weights.
    Scans data_dir for video files (.mp4, .avi, .mov) organised into per-trick subdirectories
    matching SKATEBOARD_TRICKS. Videos are split per class so that the training/validation ratio
    is approximately train_split for every class. DataLoaders use pinned memory, persistent workers,
    and prefetching for low-latency GPU feeding. Inverse-frequency class weights normalised to
    mean=1 are computed from the training split and returned as a CUDA tensor for use in a
    weighted CrossEntropyLoss. Returns (train_loader, val_loader, class_weights_tensor).
    """

    transform = fast_frame_transform

    video_paths = []
    labels = []

    for trick_idx, trick in enumerate(SKATEBOARD_TRICKS):
        trick_dir = os.path.join(data_dir, trick)
        if os.path.exists(trick_dir):
            video_files = glob.glob(os.path.join(trick_dir, "*.mp4")) + \
                         glob.glob(os.path.join(trick_dir, "*.avi")) + \
                         glob.glob(os.path.join(trick_dir, "*.mov"))

            for video_file in video_files:
                video_paths.append(video_file)
                labels.append(trick_idx)

    if len(video_paths) == 0:
        raise ValueError("No video files found in the dataset directory!")

    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    for class_idx in range(len(SKATEBOARD_TRICKS)):
        class_indices = [i for i, y in enumerate(labels) if y == class_idx]
        if not class_indices:
            continue
        rng_indices = np.random.permutation(class_indices)
        split_point = int(len(rng_indices) * train_split)
        train_idx = rng_indices[:split_point]
        val_idx = rng_indices[split_point:]
        for i in train_idx:
            train_paths.append(video_paths[i])
            train_labels.append(labels[i])
        for i in val_idx:
            val_paths.append(video_paths[i])
            val_labels.append(labels[i])

    if len(train_paths) == 0 or len(val_paths) == 0:
        print("WARNING: One of the splits is empty, using all data for training")
        train_paths = video_paths
        train_labels = labels
        val_paths = video_paths[:1]
        val_labels = labels[:1]

    train_dataset = OptimizedVideoDataset(train_paths, train_labels, sequence_length, transform)
    val_dataset = OptimizedVideoDataset(val_paths, val_labels, sequence_length, transform)

    num_workers = min(4, os.cpu_count() or 1)
    persistent = True if num_workers > 0 and (os.cpu_count() or 1) > 1 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=True,
        prefetch_factor=2 if (os.cpu_count() or 1) > 1 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        drop_last=False,
        prefetch_factor=2 if (os.cpu_count() or 1) > 1 else None
    )

    class_counts = np.zeros(len(SKATEBOARD_TRICKS), dtype=np.float32)
    for y in train_labels:
        class_counts[y] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_freq = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    if inv_freq.sum() > 0:
        class_weights = inv_freq * (len(SKATEBOARD_TRICKS) / max(inv_freq.sum(), 1e-8))
    else:
        class_weights = np.ones(len(SKATEBOARD_TRICKS), dtype=np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    return train_loader, val_loader, class_weights_tensor


class AttentionLayer(nn.Module):
    """Lightweight additive attention over a sequence of feature vectors.
    Learns a scalar importance score for every time step via a two-layer MLP with a Tanh
    non-linearity, then computes a numerically-stable softmax across the time dimension.
    Weights are clamped to [1e-8, 1] to prevent underflow before being used to scale the
    input sequence element-wise. Returns the attended sequence and the attention weight tensor.
    """

    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        attention_weights = attention_weights - attention_weights.max(dim=1, keepdim=True).values
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_weights = torch.clamp(attention_weights, min=1e-8, max=1.0)
        attended_output = x * attention_weights
        return attended_output, attention_weights


class TrickRecognitionModel(nn.Module):
    """End-to-end video classification model combining a ResNet18 frame encoder,
    a bidirectional LSTM temporal model, additive attention pooling, and a two-layer
    MLP classifier. For a batch of shape (B, T, C, H, W) the backbone processes all
    T frames in parallel by reshaping to (B*T, C, H, W), producing 512-dimensional
    frame embeddings. The LSTM then models temporal dependencies across the sequence,
    and the attention mechanism weights each time step's contribution before summation
    into a single fixed-size representation fed to the classifier head. Xavier
    initialisation is applied to all linear and LSTM weight matrices.
    """

    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        base.fc = nn.Identity()
        self.base = base

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier uniform initialisation to all Linear weight matrices and zero-initialise
        all biases. For LSTM modules, Xavier uniform is applied to weight tensors and biases are
        zeroed, following the same naming convention used by PyTorch (weight_ih, weight_hh, bias_ih,
        bias_hh). This improves gradient flow at the start of training and reduces the risk of
        saturated activations or exploding gradients in the recurrent layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.base(x)
        feats = feats.view(B, T, 512)

        lstm_out, _ = self.lstm(feats)

        att_weights = self.attention(lstm_out)
        att_weights = F.softmax(att_weights, dim=1)
        attended = lstm_out * att_weights

        pooled = attended.sum(dim=1)

        out = self.fc(pooled)
        return out, att_weights.squeeze(-1)


class DataPrefetcher:
    """Asynchronous data prefetcher that overlaps host-to-device transfers with GPU compute.
    Uses a dedicated CUDA stream to move the next batch to the GPU while the current batch
    is being processed, eliminating PCIe transfer latency from the critical training path.
    Iteration follows a preload-then-consume pattern: the constructor loads the first batch
    and each call to next() returns the ready batch while simultaneously loading the following
    one. Returns (None, None) once the underlying DataLoader is exhausted.
    """

    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        target = self.next_target
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return data, target


class OptimizedTrainer:
    """Full training pipeline with automatic mixed precision, gradient accumulation, and early stopping.
    Wraps a TrickRecognitionModel with an AdamW optimiser, a StepLR scheduler, and AMP GradScaler.
    Training uses gradient accumulation over two micro-batches to maintain a larger effective batch
    size without additional memory cost. NaN/Inf values in model outputs and losses are detected and
    corrected in-place rather than skipping batches, preserving stable loss curves. Validation runs
    without gradient computation and records per-epoch confusion matrices. The best checkpoint
    (by validation accuracy) is saved to disk; training halts automatically when no improvement is
    observed for patience consecutive epochs.
    """

    def __init__(self, model, train_loader, val_loader, device, learning_rate=1e-3, class_weights: torch.Tensor = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.scaler = GradScaler()

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3, eps=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

        self.max_grad_norm = 1.0

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.confusion_matrices = []

    def train_epoch(self):
        """Execute one full pass over the training DataLoader with mixed precision and gradient accumulation.
        Gradients are accumulated over accumulation_steps batches before a parameter update, doubling
        the effective batch size. Input tensors are sanitised with nan_to_num before the forward pass.
        Abnormal model outputs are clamped rather than discarded so every batch contributes to learning.
        Returns the average cross-entropy loss and top-1 accuracy (%) across all batches in the epoch.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        accumulation_steps = 2
        self.optimizer.zero_grad()

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False, ncols=120,
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

            with autocast():
                output, _ = self.model(data)
                if torch.isnan(output).any() or torch.isinf(output).any():
                    output = torch.clamp(torch.nan_to_num(output, nan=0.0), min=-10.0, max=10.0)
                loss = self.criterion(output, target) / accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    safe_output = torch.clamp(output.detach(), min=-10.0, max=10.0)
                    loss = self.criterion(safe_output, target) / accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 5 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.4f}',
                    'Acc': f'{100. * correct / total:.2f}%',
                    'GPU': f'{torch.cuda.memory_allocated(0) / 1024**3:.1f}GB'
                })

        progress_bar.close()
        avg_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0

        return avg_loss, accuracy

    def validate_epoch(self):
        """Evaluate the model on the full validation DataLoader without gradient computation.
        Runs under torch.no_grad() and autocast() for efficiency. Abnormal outputs are
        clamped identically to the training path to keep loss values comparable. Accumulates
        per-sample predictions and ground-truth labels to compute a full confusion matrix at
        the end of the epoch. Returns (avg_loss, accuracy_percent, confusion_matrix_ndarray).
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating", leave=False, ncols=120,
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                with autocast():
                    output, _ = self.model(data)
                    if torch.isnan(output).any() or torch.isinf(output).any():
                        output = torch.clamp(torch.nan_to_num(output, nan=0.0), min=-10.0, max=10.0)
                    loss = self.criterion(output, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                        safe_output = torch.clamp(output.detach(), min=-10.0, max=10.0)
                        loss = self.criterion(safe_output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        cm = confusion_matrix(all_targets, all_preds)
        self.confusion_matrices.append(cm)

        return avg_loss, accuracy, cm

    def train(self, num_epochs=50, save_path="optimized_skateboard_model.pth", patience=8):
        """Run the complete training loop for up to num_epochs epochs with early stopping.
        Each epoch calls train_epoch() then validate_epoch(), steps the LR scheduler, appends
        metrics to history lists, and flushes the GPU cache. When validation accuracy improves
        a full checkpoint (model weights, optimiser state, scheduler state, and all history) is
        written to save_path. If patience consecutive epochs pass without improvement the loop
        terminates early. Returns (train_losses, val_losses, train_accuracies, val_accuracies).
        """
        best_val_acc = 0
        patience_counter = 0
        best_epoch = 0

        print(f"Starting OPTIMIZED training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Early stopping patience: {patience} epochs")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("=" * 60)
            print(f"Progress: {epoch+1}/{num_epochs} epochs ({((epoch+1)/num_epochs)*100:.1f}%)")

            train_loss, train_acc = self.train_epoch()

            val_loss, val_acc, cm = self.validate_epoch()

            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            epoch_time = time.time() - epoch_start

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Epoch Time: {epoch_time:.2f}s")
            print(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch+1})")

            allocated, reserved, total, utilization = monitor_gpu()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'best_val_acc': best_val_acc,
                    'best_epoch': best_epoch,
                    'confusion_matrix': cm,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_accuracies': self.val_accuracies,
                    'confusion_matrices': self.confusion_matrices
                }

                torch.save(checkpoint, save_path)
                print(f"NEW BEST MODEL SAVED! Val Acc: {val_acc:.2f}%")

            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                print(f"\nEARLY STOPPING TRIGGERED!")
                print(f"No improvement for {patience} consecutive epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
                break

        total_time = time.time() - start_time
        print(f"\nTRAINING COMPLETED!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch+1})")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best model saved as: {save_path}")

        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path="training_history.png"):
    """Produce and save a 2×2 grid of training diagnostic plots.
    The four subplots show: (1) train vs validation loss, (2) train vs validation accuracy,
    (3) training loss alone over time, and (4) validation accuracy alone over time. All axes
    include a light grid for readability. The figure is saved to save_path at 300 dpi and
    also displayed interactively.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(train_losses, label='Train Loss', color='blue')
    ax3.set_title('Training Loss Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)

    ax4.plot(val_accs, label='Val Acc', color='red', linewidth=2)
    ax4.set_title('Validation Accuracy Improvement')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    """Render a labelled heatmap of the confusion matrix and save it to disk.
    Uses seaborn with the Blues colormap and annotates each cell with its integer count.
    Trick names from SKATEBOARD_TRICKS are used as axis tick labels; x-axis labels are
    rotated 45° for readability. Saved at 300 dpi to save_path and shown interactively.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SKATEBOARD_TRICKS, yticklabels=SKATEBOARD_TRICKS)
    plt.title('Confusion Matrix - Skateboard Trick Detection')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_classification_report(cm, save_path="classification_report.txt"):
    """Derive per-class and macro-averaged precision, recall, and F1 from a confusion matrix.
    Metrics are computed directly from the confusion matrix using vectorised NumPy operations
    with divide-by-zero protection; classes with no predicted or no actual samples receive a
    score of 0.0. Results are formatted into a human-readable text report that includes a
    per-trick breakdown and overall macro averages, written to save_path and also printed to
    stdout. Returns the formatted report string.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(np.diag(cm), np.sum(cm, axis=0), where=np.sum(cm, axis=0)!=0)
        recall = np.divide(np.diag(cm), np.sum(cm, axis=1), where=np.sum(cm, axis=1)!=0)
        f1_score = np.divide(2 * (precision * recall), (precision + recall), where=(precision + recall)!=0)

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)

    report = f"""
SKATEBOARD TRICK DETECTION - CLASSIFICATION REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

TRICK-WISE PERFORMANCE:
"""

    for i, trick in enumerate(SKATEBOARD_TRICKS):
        report += f"""
{trick}:
  Precision: {precision[i]:.4f}
  Recall:    {recall[i]:.4f}
  F1-Score:  {f1_score[i]:.4f}
"""

    report += f"""
OVERALL METRICS:
  Macro Precision: {float(np.mean(precision)):.4f}
  Macro Recall:    {float(np.mean(recall)):.4f}
  Macro F1-Score:  {float(np.mean(f1_score)):.4f}
  Overall Accuracy: {np.trace(cm) / np.sum(cm):.4f}
"""

    with open(save_path, 'w') as f:
        f.write(report)

    print(report)
    return report


class ConfidencePredictor:
    """Inference wrapper that loads a saved TrickRecognitionModel checkpoint and predicts
    skateboard tricks from video files with detailed confidence analysis. Videos are decoded
    with OpenCV, resampled to a fixed frame count, and preprocessed with standard ImageNet
    normalisation before inference. The forward pass runs under autocast() for consistency
    with training. Softmax probabilities are converted into a ranked list of (trick, confidence)
    pairs, accompanied by a confidence_analysis dict containing max confidence, the gap to the
    runner-up, Shannon entropy over the full distribution, a binary is_confident flag, and a
    categorical uncertainty_level string.
    """

    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Instantiate a TrickRecognitionModel and load weights from a saved checkpoint file.
        If the file does not exist, the model is returned with random initialisation and a
        warning is printed. The loaded model is set to eval mode and moved to self.device
        before being returned.
        """
        model = TrickRecognitionModel(num_classes=15)

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            print(f"Model file {model_path} not found. Using untrained model.")

        model.eval()
        return model.to(self.device)

    def predict_with_confidence(self, video_path, top_k=5):
        """Run inference on a single video and return ranked predictions with confidence metrics.
        Preprocesses the video via preprocess_video(), performs a no-gradient forward pass, and
        converts raw logits to softmax probabilities. Returns three values: a list of top_k dicts
        each containing 'trick', 'confidence' (%), and 'index'; a NumPy array of per-frame attention
        weights; and a confidence_analysis dict with max_confidence, confidence_gap, entropy,
        is_confident (bool), and uncertainty_level ('Low'/'Medium'/'High'). Returns (None, None,
        None) on any exception.
        """
        try:
            video_tensor = self.preprocess_video(video_path)

            with torch.no_grad():
                with autocast():
                    output, attention_weights = self.model(video_tensor)
                    probabilities = F.softmax(output, dim=1)

                all_probs = probabilities.cpu().numpy()[0]
                all_indices = np.argsort(all_probs)[::-1]

                top_probs = all_probs[all_indices[:top_k]]
                top_indices = all_indices[:top_k]

                max_confidence = top_probs[0] * 100
                confidence_gap = (top_probs[0] - top_probs[1]) * 100 if len(top_probs) > 1 else 0
                entropy = -np.sum(all_probs * np.log(all_probs + 1e-8))

                predictions = []
                for i in range(top_k):
                    trick_name = IDX_TO_TRICK[top_indices[i]]
                    confidence = top_probs[i] * 100
                    predictions.append({
                        'trick': trick_name,
                        'confidence': confidence,
                        'index': top_indices[i]
                    })

                confidence_analysis = {
                    'max_confidence': max_confidence,
                    'confidence_gap': confidence_gap,
                    'entropy': entropy,
                    'is_confident': max_confidence > 70 and confidence_gap > 10,
                    'uncertainty_level': 'Low' if max_confidence > 80 else 'Medium' if max_confidence > 60 else 'High'
                }

                return predictions, attention_weights.cpu().numpy()[0], confidence_analysis

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None, None, None

    def preprocess_video(self, video_path, sequence_length=16):
        """Decode and preprocess a video file into a model-ready batch tensor of shape (1, T, C, H, W).
        Opens the file with OpenCV, samples sequence_length frames at linearly-spaced positions,
        converts each frame from BGR to RGB, resizes to 224×224, and applies self.transform (ImageNet
        normalisation). Failed frame reads duplicate the previous valid frame; if the very first read
        fails a ValueError is raised. The list of processed tensors is stacked and given a batch
        dimension before being moved to self.device.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            raise ValueError("Video file is corrupted or empty")

        frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    raise ValueError("Could not read any frames from video")

        cap.release()

        processed_frames = [self.transform(frame) for frame in frames]
        video_tensor = torch.stack(processed_frames).unsqueeze(0)

        return video_tensor.to(self.device)


def monitor_gpu():
    """Query current CUDA memory statistics and print a utilisation summary.
    Reports allocated, reserved, and total GPU memory in GB, derives a utilisation
    percentage from allocated/total, and prints a status message indicating whether
    utilisation falls within the target range (60–90%), is too low, or is too high.
    Returns (allocated_GB, reserved_GB, total_GB, utilization_percent); returns
    (0, 0, 0, 0) when CUDA is not available.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        utilization = (allocated / total) * 100

        print(f"GPU Memory: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB (allocated/reserved/total)")
        print(f"GPU Utilization: {utilization:.1f}%")

        if 60 <= utilization <= 90:
            print(f"GPU utilization is GOOD at {utilization:.1f}% (target: 60-90%)!")
        elif utilization < 60:
            print(f"WARNING: Low GPU utilization at {utilization:.1f}%! Consider increasing batch size or model complexity.")
        else:
            print(f"WARNING: High GPU utilization at {utilization:.1f}%! Consider reducing batch size.")

        return allocated, reserved, total, utilization
    return 0, 0, 0, 0


def main():
    """Entry point for the full training pipeline.
    Calls optimize_gpu(), scans the Tricks/ directory for labelled video files,
    constructs DataLoaders via create_optimized_data_loaders(), instantiates
    TrickRecognitionModel and OptimizedTrainer, runs training for up to 50 epochs
    with early stopping (patience=8), then generates training history plots, a
    confusion matrix heatmap, a classification report, and a sample confidence
    prediction on the first available .mov file. All output files are saved to the
    current working directory.
    """
    print("STEADY GPU SKATEBOARD TRICK DETECTION (BASELINE)")
    print("="*60)
    print("Target: Steady 90% GPU utilization")
    print("GPU: GTX 1660 SUPER 6GB")
    print("Model: ResNet18 + LSTM + Attention (Advanced)")
    print("Optimized for consistent GPU usage")
    print("="*60)
    optimize_gpu()

    data_directory = "Tricks"

    video_count = 0
    for trick in SKATEBOARD_TRICKS:
        trick_dir = os.path.join(data_directory, trick)
        if os.path.exists(trick_dir):
            video_files = glob.glob(os.path.join(trick_dir, "*.mp4")) + \
                         glob.glob(os.path.join(trick_dir, "*.avi")) + \
                         glob.glob(os.path.join(trick_dir, "*.mov"))
            video_count += len(video_files)

    print(f"Total videos found: {video_count}")

    if video_count == 0:
        print("No videos found! Please add videos to sample_skateboard_data directory")
        return

    print("\nCreating optimized data loaders...")
    train_loader, val_loader, class_weights = create_optimized_data_loaders(
        data_dir=data_directory,
        batch_size=16 if torch.cuda.is_available() else 8,
        sequence_length=16,
        train_split=0.8
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("\nCreating baseline model...")
    model = TrickRecognitionModel(num_classes=15).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = OptimizedTrainer(
        model, train_loader, val_loader, device, learning_rate=1e-3, class_weights=class_weights
    )

    print("\nStarting optimized training...")
    train_losses, val_losses, train_accs, val_accs = trainer.train(
        num_epochs=50,
        save_path="baseline_skateboard_model.pth",
        patience=8
    )

    print("\nGenerating visualizations...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    if trainer.confusion_matrices:
        plot_confusion_matrix(trainer.confusion_matrices[-1])
        generate_classification_report(trainer.confusion_matrices[-1])

    print("\nTesting confidence prediction...")
    predictor = ConfidencePredictor("baseline_skateboard_model.pth", device)

    test_video = None
    for trick in SKATEBOARD_TRICKS:
        trick_dir = os.path.join(data_directory, trick)
        if os.path.exists(trick_dir):
            video_files = glob.glob(os.path.join(trick_dir, "*.mov"))
            if video_files:
                test_video = video_files[0]
                break

    if test_video:
        predictions, attention, confidence = predictor.predict_with_confidence(test_video)
        if predictions:
            print(f"\nCONFIDENCE ANALYSIS for: {os.path.basename(test_video)}")
            print("="*50)
            for i, pred in enumerate(predictions):
                print(f"{i+1}. {pred['trick']}: {pred['confidence']:.2f}%")
            print(f"\nConfidence Metrics:")
            print(f"   Max Confidence: {confidence['max_confidence']:.2f}%")
            print(f"   Confidence Gap: {confidence['confidence_gap']:.2f}%")
            print(f"   Uncertainty: {confidence['uncertainty_level']}")

    print("\nBASELINE TRAINING COMPLETED!")
    print("Files saved:")
    print("  - baseline_skateboard_model.pth (model)")
    print("  - training_history.png (plots)")
    print("  - confusion_matrix.png (confusion matrix)")
    print("  - classification_report.txt (detailed report)")


if __name__ == "__main__":
    main()
