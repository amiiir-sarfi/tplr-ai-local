# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Standard library
import os
import time
import random
import asyncio
import argparse
from datetime import datetime
from collections import defaultdict
import logging
import math
import psutil
import json

# Third party
import torch
import bittensor as bt
import numpy as np
from torch.optim import AdamW, SGD
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)

# Local
import tplr
import tplr.sharded_dataset

# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Timer:
    """Context manager for timing code blocks."""
    
    _timings = defaultdict(list)
    _active_timers = {}
    _disable = False
    
    def __init__(self, name, logger=None, disabled=False, enabled=False):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.disabled = disabled or (Timer._disable and not enabled)
        
    def __enter__(self):
        if self.disabled:
            return self
        
        self.start_time = time.perf_counter()
        Timer._active_timers[self.name] = self.start_time
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disabled or self.start_time is None:
            return
            
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        Timer._timings[self.name].append(duration)
        
        if self.logger and self.name in Timer._active_timers:
            self.logger.debug(f"{self.name}: {duration:.6f}s")
            
        if self.name in Timer._active_timers:
            del Timer._active_timers[self.name]
    
    @classmethod
    def get_stats(cls, name=None):
        """Get timing statistics for a specific timer or all timers."""
        if name is not None:
            times = cls._timings.get(name, [])
            if not times:
                return {}
            return {
                'total': sum(times),
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'last': times[-1]
            }
        else:
            return {name: cls.get_stats(name) for name in cls._timings.keys()}
    
    @classmethod
    def reset(cls):
        """Reset all timings."""
        cls._timings = defaultdict(list)
        cls._active_timers = {}
    
    @classmethod
    def disable(cls, disabled=True):
        """Disable all timers."""
        cls._disable = disabled
        
    @classmethod
    def summarize(cls, logger=None):
        """Summarize all timings."""
        result = {}
        for name, times in cls._timings.items():
            if not times:
                continue
            
            stats = cls.get_stats(name)
            msg = (f"{name} - total: {stats['total']:.3f}s, "
                  f"mean: {stats['mean']:.3f}s, "
                  #   f"min: {stats['min']:.3f}s, " # Can add it back, not super useful usually
                  f"max: {stats['max']:.3f}s")
            
            result[name] = stats
            
            if logger:
                logger.info(msg)
                
        return result

def dict_parser_type(value):
    """Helper function to parse a JSON string into a dict for argparse."""
    try:
        value = value.replace("'", '"') 
        loaded_dict = json.loads(value)
        return loaded_dict 
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON format for dictionary: {value}")

class AdamBaseline:
    """
    Baseline training implementation using AdamW and DDP for comparison with
    gradient compression and peer training.
    """
    
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='AdamW DDP Baseline')
        parser.add_argument('--project', type=str, default='boom', help='Wandb project.')
        parser.add_argument('--run_name', type=str, default='', help='Wandb run name.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--hparams_file', type=str, default='hparams.json', help='hparams file.')
        
        # DDP specific args
        parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), 
                            help='Number of GPUs to use for distributed training')
        
        # Strategy args
        parser.add_argument('--strategy', type=str, default='diloco', choices=['normal', 'diloco'],
                            help='Training strategy to use (normal or diloco)')
        parser.add_argument("--anomalies", type=dict_parser_type, default={},
                            help='Dictionary of anomaly configs where keys are the worker idx and values are dictionary of anomalous configs, e.g., {"2": {"val_multiplier": 10}}')
        parser.add_argument('--grad_val_multiplier', type=int, default=1,
                            help='Multiplier for gradient vals post compression before communication (to simulate anomalies)')

        # Optimizer args
        parser.add_argument('--micro_batch_size', type=int, default=-1, 
                            help='Micro batches for data loader')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='Batch size for grad accum')
        parser.add_argument('--sequence_length', type=int, default=2048,
                            help='sequence length for training')    
        parser.add_argument('--weight_decay', type=float, default=0.1,
                            help='Weight decay for regularization')
        parser.add_argument('--warmup_steps', type=float, default=250,
                            help='Number of warmup steps for learning rate scheduler')    
        
        ## Inner optimizer
        parser.add_argument('--inner_steps', type=int, default=10,
                            help='Local steps before communication (H)')
        parser.add_argument('--inner_learning_rate', type=float, default=6e-4,
                            help='Learning rate for inner optimizer')
        parser.add_argument('--inner_optimizer', type=str, default=None, choices=['adamw'],
                            help='inner optimizer to use. None means simple gradient accumulation')
        
        ## Outer optimizer
        parser.add_argument('--outer_learning_rate', type=float, default=0.7,
                            help='Learning rate for outer optimizer')
        parser.add_argument('--outer_momentum', type=float, default=0.0,
                            help='Momentum for outer optimizer')
        parser.add_argument('--outer_nesterov', action='store_true',
                            help='Nesterov for outer optimizer')
        parser.add_argument('--outer_use_sign', type=int, default=1, choices=[0, 1],
                            help='Use sign for outer optimizer')
        parser.add_argument('--outer_optimizer', type=str, default='demo', choices=['adamw', 'demo', 'nesterov'],
                            help='Outer optimizer to use for training (adamw or demo or nesterov)')
        
        # DeMo specific args
        parser.add_argument('--compression_decay', type=float, default=0.999,
                            help='Compression decay for DeMo optimizer')
        parser.add_argument('--compression_topk', type=int, default=32,
                            help='Compression topk for DeMo optimizer')
        parser.add_argument('--compression_chunk', type=int, default=64,
                            help='Compression chunk size for DeMo optimizer')
        parser.add_argument('--use_grad_normalization', action='store_true',
                            help='Use gradient normalization for DeMo optimizer')
        parser.add_argument('--use_quantization', action='store_true',
                            help='Use quantization for DeMo optimizer')
        parser.add_argument('--quantization_bins', type=int, default=256,
                            help='Number of quantization bins')
        parser.add_argument('--quantization_range', type=int, default=6,
                            help='Quantization range in standard deviations')
        # Dataset args
        parser.add_argument('--token_budget', type=int, default=15728640,
                            help='Token budget for training. If negative, is set from hparams file.')
        parser.add_argument('--shards_path', type=str, default='~/datasets/edu_fineweb_score2_10B_tokenized_llama2',
                            help='Path to the dataset shards.')
        parser.add_argument('--shard_token_size', type=int, default=100e6,
                            help='Number of tokens in each stored shard.')
        parser.add_argument('--max_steps', type=int, default=-1,
                            help='Maximum number of training steps (None for unlimited)')
        parser.add_argument('--seed', type=str, default="adam_baseline",
                            help='Seed for deterministic page selection')
        parser.add_argument('--data_in_gpu', action='store_true',
                            help='Keep whole dataset in GPU.')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='Number of workers per DDP process for data loading')
        parser.add_argument('--num_prefetch_batches', type=int, default=0,
                            help='Number of batches to prefetch for data loading')

        # Checkpoint args
        parser.add_argument('--save_path', type=str, default='./checkpoints', 
                            help='Path to save model checkpoints')
        parser.add_argument('--save_interval', type=int, default=500, 
                            help='Save checkpoint every N windows')
        parser.add_argument('--load_checkpoint', type=str, default=None,
                            help='Path to checkpoint file to resume training from')
        
        # Timing args
        parser.add_argument('--timing_log', type=str, default='timings.log',
                           help='File to write timing information to')
        
        # torch.compile args
        parser.add_argument('--use_compile', action='store_true',
                           help='Use torch.compile to optimize model execution')
        
        bt.logging.add_args(parser)
        config = bt.config(parser)
        
        if config.strategy == "normal":
            config.inner_optimizer = None
        elif config.strategy == "diloco":
            config.inner_optimizer = 'adamw'
        
        assert config.strategy == "normal" or config.inner_optimizer is not None, "Inner optimizer must be specified if strategy is not normal"

        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
            
        return config
    
    def __init__(self):
        tplr.logger.debug("Starting AdamW baseline initialization...")
        
        self.config = AdamBaseline.config()
        hparams_file = os.path.expandvars(os.path.expanduser(self.config.hparams_file))
        self.hparams = tplr.load_hparams(hparams_file)

        if self.config.micro_batch_size < 0:
            self.config.micro_batch_size = self.config.batch_size

        self._setup_distributed()

        for worker_idx, anomaly_config in self.config.anomalies.items():
            if self.global_rank == int(worker_idx):
                for anomaly_k,anomaly_v in anomaly_config.items():
                    if hasattr(self.config, anomaly_k): # CHECKING
                        original_value = getattr(self.config, anomaly_k)
                        setattr(self.config, anomaly_k, anomaly_v) # Using setattr here
                        tplr.logger.info(f"[Rank {self.global_rank}]: Updated self.config.{anomaly_k} from {original_value} to {anomaly_v}")
                    else:
                        raise ValueError(f"Anomaly config key '{anomaly_k}' not found in config")

        self._calculate_steps()

        self._initialize_model_and_tokenizer()

        self._setup_optimizers_and_schedulers()

        self._initialize_state_and_metrics()
        
        self._initialize_dataloader()
        
        self._setup_wandb_and_logging()
        
        self._initialize_strategy()
        
        # summary info
        if self.global_rank == 0:
            
            # Calculate expected training time and tokens
            tokens_per_step = self.config.batch_size * self.world_size * self.config.sequence_length * self.config.inner_steps
            total_tokens = tokens_per_step * self.config.max_steps
            
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
                        
            tplr.logger.info("\n" + "=" * 80)
            tplr.logger.info(f"TRAINING CONFIGURATION SUMMARY:")
            tplr.logger.info(f"→ Hardware: {self.world_size} GPU(s)")
            tplr.logger.info(f"→ Model memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved (excluding batches)")
            tplr.logger.info(f"→ Training strategy: {self.config.strategy.upper()} with {self.config.inner_steps} inner steps")
            
            if self.config.strategy.lower() == "diloco":
                tplr.logger.info(f"→ Inner optimizer: {self.config.inner_optimizer} (lr={self.config.inner_learning_rate}, weight_decay={self.config.weight_decay}, inner_steps={self.config.inner_steps})")
            
            tplr.logger.info(f"→ Outer optimizer: {self.config.outer_optimizer} (lr={self.config.outer_learning_rate}, weight_decay={self.outer_weight_decay})")            
            tplr.logger.info(f"→ Batch hierarchy: {self.config.micro_batch_size} (micro) → {self.config.batch_size} (accum)")
            tplr.logger.info(f"→ Sequence length: {self.config.sequence_length} tokens per sample")
            
            # Add token computation information
            inner_effective_tokens = self.config.batch_size * self.world_size * self.config.inner_steps * self.config.sequence_length
            tplr.logger.info(f"→ Inner cycle: {inner_effective_tokens:,} tokens processed per full inner cycle across all GPUs")
            tplr.logger.info(f"→ Training plan: {self.config.max_steps:,} steps, targeting {total_tokens:,} tokens total (given target: {self.config.token_budget:,})")
            tplr.logger.info(f"→ Scheduler plan: {self.warmup_steps:,} warmup steps, {self.cosine_steps:,} cosine steps, {self.total_scheduler_steps:,} total scheduler steps")
            tplr.logger.info(f"→ Data: {len(self.train_loader.dataset)} samples with {self.config.sequence_length:,} tokens each (seq_len)")
            
            if self.config.use_compile:
                tplr.logger.info(f"→ Optimization: Using torch.compile for model execution")
            
            tplr.logger.info("=" * 80 + "\n")
    
    def _setup_distributed(self):
        """Set up the distributed training environment."""
        if self.config.local_rank != -1:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"]) 
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1

        Timer.disable(not (self.config.debug and self.global_rank == 0))
        
        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            init_process_group(backend="nccl", rank=self.global_rank, world_size=self.world_size)
            tplr.logger.info(f"Initialized DDP: rank {self.global_rank}/{self.world_size-1} on device {self.local_rank}")
            
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
    
    def _calculate_steps(self):
        """Calculate training steps."""
        if self.config.strategy.lower() == "normal":
            self.config.inner_steps = 1

        # Calculate max_steps
        if self.config.max_steps == -1:
            self.config.max_steps = (self.config.token_budget // 
                                  (self.config.batch_size * self.config.sequence_length * self.config.inner_steps * self.world_size))
        
        # Calculate total steps for schedulers
        self.total_scheduler_steps = self.config.token_budget // (self.config.batch_size * self.config.sequence_length * self.world_size) 
    
    def _initialize_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.tokenizer = self.hparams.tokenizer
        
        if self.config.debug and self.global_rank == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            tplr.logger.info(f"using model config: {self.hparams.model_config}")
            tplr.logger.info(f"→ Total params:     {total_params:,}")
            tplr.logger.info(f"→ Trainable params: {trainable_params:,}")
            
            total_params_noncausal = sum(p.numel() for p in self.model.model.parameters())
            trainable_params_noncausal = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
            tplr.logger.debug(f"→ Total params (non-causal):     {total_params_noncausal:,}")
            tplr.logger.debug(f"→ Trainable params (non-causal): {trainable_params_noncausal:,}")

        self.model.to(device=self.device)

        # Synchronize model parameters across processes for safety
        if self.world_size > 1:
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            if self.global_rank == 0:
                tplr.logger.info("Synchronized model parameters across all processes")

        if self.config.use_compile:
            self.model = torch.compile(self.model, dynamic=True) 
                
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
    
    def _initialize_dataloader(self):
        """Initialize the data loader."""
        if self.global_rank == 0:
            # Log memory before dataset creation
            ram_before = psutil.virtual_memory()
            tplr.logger.info(f"RAM before dataset creation: {ram_before.used / 1024**3:.2f}GB used, "
                           f"{ram_before.available / 1024**3:.2f}GB available")
        
        train_dataset = tplr.sharded_dataset.ShardedGPUDataset(
            shards_path=os.path.expandvars(os.path.expanduser(self.config.shards_path)),
            token_budget=self.config.token_budget,
            sequence_length=self.config.sequence_length,
            rank=self.global_rank,
            world_size=self.world_size,
            device=self.device,
            shard_token_size=self.config.shard_token_size,
            split="train",
            reside_in_gpu=self.config.data_in_gpu
        )
        
        if self.global_rank == 0:
            # Log memory after dataset creation
            ram_after = psutil.virtual_memory()
            tplr.logger.info(f"RAM after dataset creation: {ram_after.used / 1024**3:.2f}GB used, "
                           f"{ram_after.available / 1024**3:.2f}GB available, "
                           f"delta: +{(ram_after.used - ram_before.used) / 1024**3:.2f}GB")
        
        # prefetch_batches = self.config.inner_steps if self.config.strategy.lower() == "diloco" else 2
        self.train_loader = tplr.get_sharded_gpu_dataloader(
            train_dataset, 
            batch_size=self.config.micro_batch_size, 
            shuffle=True,
            num_workers=self.config.num_workers,
            num_prefetch_batches=self.config.num_prefetch_batches
        )

    def _create_scheduler(self, optimizer, lr):
        """Create a standard scheduler with warmup and cosine annealing."""
        warmup_steps = self.config.warmup_steps
        # If warmup_steps is given as a fraction of total steps:
        if warmup_steps < 1:
            warmup_steps = self.total_scheduler_steps * warmup_steps
        
        warmup_steps = int(warmup_steps)
        cosine_steps = max(1, self.total_scheduler_steps - warmup_steps)
        self.warmup_steps = warmup_steps
        self.cosine_steps = cosine_steps

        if warmup_steps >= self.total_scheduler_steps:
            raise ValueError(
                f"Warmup steps ({self.config.warmup_steps:,}) must be less than total scheduler steps "
                f"({self.total_scheduler_steps:,})."
            )
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=lr * 0.1,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    
    def _setup_optimizers_and_schedulers(self):
        """Set up optimizers and schedulers for training."""
        # Initialize inner optimizer (for Diloco)
        self.inner_optimizer = None
        if self.config.strategy == "diloco":
            if self.config.inner_optimizer.lower() == 'adamw':
                self.inner_optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.inner_learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=(0.9, 0.95)
                )
                if self.global_rank == 0:
                    tplr.logger.info(f"Using AdamW as inner optimizer with lr={self.config.inner_learning_rate} and weight_decay={self.config.weight_decay}")
            else:
                raise NotImplementedError(f"Unknown inner optimizer: {self.config.inner_optimizer}")
        
        # Initialize outer optimizer
        self.outer_weight_decay = self.config.weight_decay if self.config.strategy.lower() == "normal" else 0.0
        if self.config.outer_optimizer.lower() == 'demo':
            self.outer_optimizer = tplr.DeMo(
                self.model.parameters(),
                lr=self.config.outer_learning_rate,
                momentum=self.config.outer_momentum,
                nesterov=self.config.outer_nesterov,
                weight_decay=self.outer_weight_decay,
                compression_decay=self.config.compression_decay,
                compression_topk=self.config.compression_topk,
                compression_chunk=self.config.compression_chunk,
                use_grad_normalization=self.config.use_grad_normalization,
                use_quantization=self.config.use_quantization,
                quantization_bins=self.config.quantization_bins,
                quantization_range=self.config.quantization_range,
                use_sign=bool(self.config.outer_use_sign),
                grad_val_multiplier=self.config.grad_val_multiplier,
                process_group=dist.group.WORLD if self.world_size > 1 else None
            )
        elif self.config.outer_optimizer.lower() == 'adamw':
            self.outer_optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                betas=(0.9, 0.95),
                eps=0.1
            )
        elif self.config.outer_optimizer.lower() == 'nesterov':
            self.outer_optimizer = SGD(
                self.model.parameters(),
                lr=self.config.outer_learning_rate,
                weight_decay=self.outer_weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise NotImplementedError(f"Unknown outer optimizer: {self.config.outer_optimizer}")

        if self.global_rank == 0:
            tplr.logger.info(f"Using {self.config.outer_optimizer} outer_optimizer with DDP with LR={self.config.outer_learning_rate} and weight_decay={self.outer_weight_decay}")

        # Create scheduler
        optimizer_for_scheduler = self.inner_optimizer if self.config.strategy.lower() == "diloco" else self.outer_optimizer
        lr_for_scheduler = self.config.inner_learning_rate if self.config.strategy.lower() == "diloco" else self.config.outer_learning_rate
        scheduler = self._create_scheduler(optimizer_for_scheduler, lr_for_scheduler)
        self.scheduler = scheduler
        if self.config.strategy.lower() == "diloco":
            self.inner_scheduler = scheduler
            self.outer_scheduler = None  # No outer scheduler for Diloco
        else:
            self.inner_scheduler = None  # No inner scheduler for SimpleAccum
            self.outer_scheduler = scheduler
    
    def _initialize_state_and_metrics(self):
        """Initialize state variables and metrics tracking."""
        if self.global_rank == 0:
            os.makedirs(self.config.save_path, exist_ok=True)
        
        self.step_counter = 0
        self.global_step = 0
        self.window_step = 0
        
        self.total_tokens_processed = 0
        self.batch_times = []
        
        if self.config.load_checkpoint is not None:
            self._load_checkpoint(self.config.load_checkpoint)
    
    def _setup_wandb_and_logging(self):
        """Set up WandB and timing loggers."""
        if self.global_rank == 0:
            self.wandb = tplr.initialize_wandb(
                run_prefix='Dist',
                uid=self.config.run_name,
                config=self.config,
                group='baseline',
                job_type='adam_training'
            )
        else:
            self.wandb = None

        self.timing_logger = None
        if self.config.debug:
            self.setup_timing_logger()
    
    def _initialize_strategy(self):
        """Initialize the training strategy."""
        if self.config.strategy.lower() == "diloco":
            self.strategy = tplr.Diloco(
                self.device, self.world_size, self.global_rank, 
                self.tokenizer, self.config
            )
        else:
            self.strategy = tplr.SimpleAccum(
                self.device, self.world_size, self.global_rank, 
                self.tokenizer, self.config
            )
    
    def setup_timing_logger(self):
        """Set up a separate logger for performance timing information."""
        log_dir = os.path.dirname(self.config.timing_log)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        self.timing_logger = logging.getLogger('timing')
        self.timing_logger.setLevel(logging.DEBUG)
        self.timing_logger.propagate = False  # Don't propagate to root logger
        
        if self.timing_logger.handlers:
            self.timing_logger.handlers.clear()
        file_handler = logging.FileHandler(self.config.timing_log, mode='w')
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.timing_logger.addHandler(file_handler)
        
        self.timing_logger.info(f"Starting new training run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.timing_logger.info(f"Configuration: optimizer={self.config.outer_optimizer}, lr={self.config.outer_learning_rate}, "
                               f"world_size={self.world_size}, batch_size={self.config.batch_size}")
        self.timing_logger.info("-" * 80)

    def log_timing(self, message):
        """Helper to log timing information to the timing log file."""
        if self.global_rank == 0 and self.timing_logger is not None:
            self.timing_logger.info(message)
            
    async def run(self):
        """Main training loop."""
        for window in range(self.window_step, self.config.max_steps):
            if self.global_step >= self.config.max_steps:
                tplr.logger.info(f"Reached maximum steps {self.config.max_steps}. Stopping.")
                break
                
            if self.global_rank == 0:
                tplr.logger.info(f"\n{'-' * 40} Window: {window} {'-' * 40}")
                if self.config.debug:
                    self.log_timing(f"Window {window} - Starting gradient accumulation")
            
            # Reset timers for this window
            if self.global_rank == 0:
                Timer.reset()
                
            with Timer("window_total", enabled=True):
                # Training loop
                if self.global_rank == 0:
                    tplr.logger.info("Start accumulating gradients...")
                
                if self.inner_optimizer:
                    self.inner_optimizer.zero_grad()
                self.outer_optimizer.zero_grad()
                self.model.zero_grad()
                
                # Use strategy for inner step (gradient accumulation)
                with Timer("inner_step"):
                    metrics = self.strategy.inner_step(
                        self.model, self.train_loader, 
                        self.inner_optimizer, self.inner_scheduler
                    )
                
                # Reduce metrics across workers
                with Timer("reduce_metrics"):
                    metrics_to_reduce = torch.tensor(
                        [metrics["total_loss"], metrics["batch_count"], metrics["batch_tokens"]], 
                        device=self.device
                    )
                    
                    if self.world_size > 1:
                        torch.distributed.all_reduce(metrics_to_reduce, op=torch.distributed.ReduceOp.SUM)
                    
                    total_loss = metrics_to_reduce[0].item()
                    batch_count = metrics_to_reduce[1].item() 
                    batch_tokens = metrics_to_reduce[2].item()
                
                # Use strategy for outer step
                with Timer("outer_step"):
                    self.strategy.outer_step(self.model, self.outer_optimizer, self.scheduler)
            
            if self.global_rank == 0:
                # Calculate tokens per second
                all_stats = Timer.summarize(logger=self.timing_logger if self.config.debug else None)
                window_duration = all_stats.get('window_total', {}).get('total', 0)

                tokens_per_second = batch_tokens / window_duration
                tplr.logger.info(f"Window {window}: Processing rate: {tokens_per_second:.2f} tokens/sec")

                timer_metrics = {}
                timer_metrics[f"timing/tokens_per_sec"] = tokens_per_second
                if self.config.debug:
                    self.log_timing(f"Window {window} - Timing summary:")
                    self.log_timing(f"  Total tokens: {batch_tokens}, Tokens/sec: {tokens_per_second:.2f}")

                    for timer_name, stats in all_stats.items():
                        timer_metrics[f"timing/{timer_name}/total"] = stats.get('total', 0)
                        timer_metrics[f"timing/{timer_name}/mean"] = stats.get('mean', 0)
                        timer_metrics[f"timing/{timer_name}/max"] = stats.get('max', 0)
                    
                    self.log_timing("-" * 40)
                
                tplr.logger.info(f"effective_batch_size: {self.config.batch_size * self.world_size}")
                tplr.logger.info(f"Window {window} completed: {batch_count} batches with {batch_tokens} tokens")
                
                # Log gradient metrics
                grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
                weight_norms = [p.norm().item() for p in self.model.parameters()]

                tplr.logger.info(
                    f"gradient/mean_grad_norm: {sum(grad_norms) / len(grad_norms) if grad_norms else 0 : 0.3f}, "
                    f"gradient/max_grad_norm: {max(grad_norms) if grad_norms else 0 : 0.3f}, "
                    f"gradient/min_grad_norm: {min(grad_norms) if grad_norms else 0 : 0.3f}, "
                    f"gradient/grad_norm_std: {torch.tensor(grad_norms).std().item() if grad_norms else 0 : 0.3f}, "
                    f"gradient/mean_weight_norm: {sum(weight_norms) / len(weight_norms) : 0.3f}"
                )
                
                # Wandb logging
                metrics_dict = {
                    # Training metrics
                    "baseline/loss": total_loss/batch_count, 
                    "baseline/total_tokens": self.total_tokens_processed + batch_tokens,
                    "baseline/batch_tokens": batch_tokens,
                    "baseline/global_step": self.global_step,
                    "baseline/perplexity": torch.exp(torch.tensor(total_loss/batch_count)).item(),
                    "baseline/tokens_per_sec": tokens_per_second,
                    
                    # Resource metrics
                    "misc/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
                    "misc/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2,  # MB
                    
                    # Network metrics
                    "setting/num_gpus": self.world_size,
                    "setting/effective_batch_size": self.world_size * self.config.batch_size * self.config.inner_steps,
                    "setting/learning_rate": self.scheduler.get_last_lr()[0],
                    
                    # Gradient statistics as points
                    "gradient/mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0,
                    "gradient/max_grad_norm": max(grad_norms) if grad_norms else 0,
                    "gradient/min_grad_norm": min(grad_norms) if grad_norms else 0,
                    "gradient/grad_norm_std": torch.tensor(grad_norms).std().item() if grad_norms else 0,
                    "gradient/mean_weight_norm": sum(weight_norms) / len(weight_norms),
                    "gradient/grad_to_weight_ratio": (sum(grad_norms) / len(grad_norms)) / (sum(weight_norms) / len(weight_norms)) if grad_norms and weight_norms else 0,
                    
                }
                
                # Add optimizer-specific learning rates
                if self.config.strategy.lower() == "diloco" and self.inner_optimizer:
                    metrics_dict["setting/inner_learning_rate"] = self.inner_scheduler.get_last_lr()[0]
                    metrics_dict["setting/outer_learning_rate"] = self.config.outer_learning_rate
                
                # Add DeMo specific metrics if using DeMo optimizer
                if self.config.outer_optimizer.lower() == 'demo':
                    metrics_dict.update({
                        "misc/data_transmit": self.outer_optimizer.data_transmit / 1024**2,  # MB
                        "misc/data_receive": self.outer_optimizer.data_receive / 1024**2,  # MB
                        "misc/communication_efficiency": batch_tokens / (self.outer_optimizer.data_transmit / 1024**2) if self.outer_optimizer.data_transmit > 0 else 0,  # tokens/MB transmitted
                    })
                
                metrics_dict.update(timer_metrics)

                self.wandb.log(metrics_dict, step=self.global_step)
                
                # Update total tokens processed
                self.total_tokens_processed += batch_tokens
                
                # Save checkpoint every save_interval windows
                if (window + 1) % self.config.save_interval == 0 or window == self.config.max_steps - 1:
                    self._save_checkpoint(window)
            else:
                self.total_tokens_processed += batch_tokens
            
            self.global_step += 1
            self.window_step += 1
            
        if self.global_rank == 0:
            tplr.logger.info(f"Training completed with {self.total_tokens_processed} tokens processed.")
                
                
    def _save_checkpoint(self, window):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.save_path, f"demo_checkpoint_window_{window}_{timestamp}.pt")
        
        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
            
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.outer_optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }
        
        # Add inner optimizer/scheduler state for Diloco
        if self.config.strategy == "diloco" and self.inner_optimizer is not None:
            checkpoint.update({
                'inner_optimizer_state_dict': self.inner_optimizer.state_dict(),
                'inner_scheduler_state_dict': self.inner_scheduler.state_dict() if self.inner_scheduler else None,
            })
            
        # Add training state
        checkpoint.update({
            'window': window,
            'global_step': self.global_step,
        })
        
        torch.save(checkpoint, path)
        tplr.logger.info(f"Saved checkpoint to {path}")
        
    def _load_checkpoint(self, checkpoint_path):
        """Load model, optimizer, and training state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            tplr.logger.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        tplr.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load optimizer and scheduler states
        self.outer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load inner optimizer and scheduler for Diloco
        if self.config.strategy == "diloco":
            if 'inner_optimizer_state_dict' in checkpoint and self.inner_optimizer:
                self.inner_optimizer.load_state_dict(checkpoint['inner_optimizer_state_dict'])
            if 'inner_scheduler_state_dict' in checkpoint and self.inner_scheduler:
                self.inner_scheduler.load_state_dict(checkpoint['inner_scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        # Resume from the next window after the saved one
        self.window_step = checkpoint.get('window', 0) + 1
        
        tplr.logger.info(f"Resumed training from window {self.window_step-1}, global step {self.global_step}")
        
    def cleanup(self):
        """Clean up resources."""
        if self.world_size > 1:
            destroy_process_group()
        
        if self.wandb is not None:
            self.wandb.finish()
            
        # Close timing logger
        if self.global_rank == 0 and self.timing_logger is not None:
            for handler in self.timing_logger.handlers:
                handler.close()
                self.timing_logger.removeHandler(handler)

async def main():
    """Entry point."""
    baseline = AdamBaseline()
    
    try:
        await baseline.run()
    except KeyboardInterrupt:
        tplr.logger.info("Training interrupted by user")
    except Exception as e:
        tplr.logger.error(f"Error during training: {e}")
        raise
    finally:
        baseline.cleanup()
            


if __name__ == "__main__":
    asyncio.run(main())