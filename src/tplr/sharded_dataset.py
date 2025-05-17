# src/tplr/in_memory_gpu_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import math
import os
import glob
import numpy as np
from pathlib import Path
from typing import Literal
import tplr

class ShardedGPUDataset(Dataset):
    """
    A PyTorch Dataset that preloads tokenized data from .npy shards into VRAM,
    splits it among DDP workers, and serves sequences directly from GPU.
    """
    def __init__(self,
                 shards_path: str,
                 token_budget: int,
                 sequence_length: int,
                 rank: int,
                 world_size: int,
                 device: torch.device,
                 shard_token_size: int = 100_000_000, # Expected tokens per .npy shard
                 split: Literal["train"] = "train"): # only supports "train" for now
        """
        Args:
            shards_path (str): Path to the directory containing .npy token shards.
            token_budget (int): Total number of tokens to be used across all workers.
            sequence_length (int): The length of each sequence to be returned.
            rank (int): The rank of the current DDP process.
            world_size (int): The total number of DDP processes.
            device (torch.device): The CUDA device for this rank (e.g., torch.device("cuda:0")).
            shard_token_size (int): Expected number of tokens in each .npy shard file.
                                    Used to calculate how many shards to load for the budget.
            split (str): Train test split to load data (e.g., "train").
        """
        super().__init__()
        self.shards_path = Path(shards_path)
        self.token_budget = token_budget
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.shard_token_size = shard_token_size
        self.shard_filename_prefix = f"{split}_"

        if not self.shards_path.is_dir():
            raise FileNotFoundError(f"Shards directory not found: {self.shards_path}")

        # 1. Discover and sort shard files
        shard_files = sorted(glob.glob(str(self.shards_path / f"{self.shard_filename_prefix}*.npy")))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found with prefix '{self.shard_filename_prefix}' in {self.shards_path}")

        # 2. Calculate how many shards to load and load them
        num_shards_to_load = math.ceil(self.token_budget / self.shard_token_size)
        
        if num_shards_to_load > len(shard_files):
            raise ValueError(f"[Rank {self.rank}]: Requested to load {num_shards_to_load} shards, but only {len(shard_files)} are available.")

        loaded_token_arrays = []
        current_loaded_tokens = 0

        for i in range(num_shards_to_load):
            shard_file_path = shard_files[i]
            # tplr.logger.debug(f"[Rank {self.rank}] Loading shard: {shard_file_path}")
            try:
                shard_data_np = np.load(shard_file_path).astype(np.int32)
                loaded_token_arrays.append(torch.tensor(shard_data_np, dtype=torch.long)) # Ensure long for token IDs
                current_loaded_tokens += len(shard_data_np)
            except Exception as e:
                raise IOError(f"Error loading shard file {shard_file_path}: {e}")

        if not loaded_token_arrays:
            raise ValueError("No tokens loaded. Check shard files or budget.")

        all_tokens = torch.cat(loaded_token_arrays, dim=0)
        
        # 3. Trim to exact token budget.
        all_tokens = all_tokens[:self.token_budget]

        # 4. Split the global data for the current worker (rank)
        num_all_tokens = len(all_tokens)
        
        worker_start_idx = self.rank * (num_all_tokens // self.world_size)
        worker_end_idx = (self.rank + 1) * (num_all_tokens // self.world_size)
        
        
        worker_tokens_cpu = all_tokens[worker_start_idx:worker_end_idx]
        
        # 5. Move this worker's data to its specified CUDA device
        self.worker_tokens_gpu = worker_tokens_cpu.to(self.device)
        
        # Calculate number of full sequences (samples) for this worker
        self.num_samples = len(self.worker_tokens_gpu) // self.sequence_length
        tplr.logger.debug(f"[Rank {self.rank}] Worker token range: {worker_start_idx} to {worker_end_idx} "
                          f"(total: {worker_end_idx-worker_start_idx}/{num_all_tokens}). Number of samples: {self.num_samples}.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sequence_length
        end = start + self.sequence_length
        
        return self.worker_tokens_gpu[start:end]


def get_sharded_gpu_dataloader(
    dataset: ShardedGPUDataset,
    batch_size: int,
    shuffle: bool = True,
):
    """
    Creates a PyTorch DataLoader for the ShardedGPUDataset.
    
    Args:
        dataset: The ShardedGPUDataset instance
        batch_size: Number of sequences per batch
        shuffle: Whether to shuffle the dataset.
    """
    if not isinstance(dataset, ShardedGPUDataset):
        raise TypeError("dataset must be an instance of ShardedGPUDataset")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )