from abc import ABC, abstractmethod
import torch
import contextlib
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import tplr

class InnerOuterStrategy(ABC):
    @abstractmethod
    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None): 
        """Execute inner optimization step"""
        pass
    
    @abstractmethod
    def outer_step(self, model, optimizer, scheduler=None): 
        """Execute outer optimization step"""
        pass

class SimpleAccum(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config, hparams):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.hparams = hparams
    
    def inner_step(self, model, loader, inner_optimizer=None, inner_scheduler=None):
        """
        Process batches from the loader until the accumulation batch size target is reached.
        Returns metrics dictionary with loss and token counts.
        """
        total_loss = 0
        batch_tokens = 0
        batch_count = 0
        accum_batch_size = 0
        
        if hasattr(model, 'no_sync') and self.world_size > 1:
            ddp_context = model.no_sync()
        else:
            ddp_context = contextlib.nullcontext()
            
        with ddp_context:
            for i, batch in enumerate(loader):
                # Check if we've reached accumulation batch size
                if accum_batch_size >= self.hparams.batch_size:
                    break
                
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                
                # Update accumulated batch size
                current_batch_size = len(batch)
                accum_batch_size += current_batch_size
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)
                
                loss = outputs.loss / self.hparams.batch_size
                loss.backward()
                
                total_loss += outputs.loss.item()
                batch_count += 1
                batch_tokens += (labels != -100).sum().item()
                
                if self.global_rank == 0 and i % 5 == 0:
                    tplr.logger.info(f'Batch {i}, loss: {outputs.loss.item():.4f}, accum: {accum_batch_size}/{self.hparams.batch_size}')
        
        # Return metrics
        return {
            "total_loss": total_loss,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
            "accum_batch_size": accum_batch_size
        }
    
    def outer_step(self, model, optimizer, scheduler):
        """
        Synchronize gradients (if DDP) and apply optimizer step
        """
        if self.world_size > 1 and self.config.outer_optimizer != "demo":
            self._synchronize_gradients(model)
        
        optimizer.step()
        scheduler.step()
    
    def _synchronize_gradients(self, model):
        """Helper method to synchronize gradients in DDP setting"""
        actual_model = model.module if hasattr(model, 'module') else model
        for param in actual_model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

class Diloco(InnerOuterStrategy):
    def __init__(self, device, world_size, global_rank, tokenizer, config, hparams):
        self.device = device
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.config = config
        self.hparams = hparams
        
        # Store offloaded parameters
        self.params_offloaded = None
    
    def inner_step(self, model, loader, inner_optimizer, inner_scheduler):
        """
        Train the model with inner optimizer for multiple steps.
        Before training, offload parameters for later use in outer step.
        """
        total_loss = 0
        batch_tokens = 0
        batch_count = 0
        
        # Offload current parameters before training
        self.params_offloaded = self._get_offloaded_param(model)
        
        if hasattr(model, 'no_sync') and self.world_size > 1:
            ddp_context = model.no_sync()
        else:
            ddp_context = contextlib.nullcontext()
        
        inner_step_count = 0
        accum_batch_size = 0
        
        with ddp_context:
            for i, batch in enumerate(loader):
                input_ids = torch.tensor(batch, dtype=torch.long).to(self.device)
                labels = input_ids.clone()
                labels = torch.where(labels == self.tokenizer.pad_token_id, -100, labels)
                
                current_batch_size = len(batch)
                accum_batch_size += current_batch_size
                
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, labels=labels)
                
                loss = outputs.loss / self.hparams.batch_size
                loss.backward()
                
                total_loss += outputs.loss.item()
                batch_count += 1
                batch_tokens += (labels != -100).sum().item()
                
                
                # If we've accumulated enough batch size, do an optimization step
                if accum_batch_size >= self.hparams.batch_size:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    inner_optimizer.step()
                    inner_scheduler.step()
                    inner_optimizer.zero_grad()
                    
                    if self.global_rank == 0 and inner_step_count % 5 == 0:
                        tplr.logger.info(f'Inner Step {inner_step_count+1}/{self.hparams.inner_steps}, '
                            f'Batch {i}, loss: {outputs.loss.item():.4f}, '
                            f'accum: {accum_batch_size}/{self.hparams.batch_size}')
                        
                    inner_step_count += 1
                    accum_batch_size = 0
                
                    # If we've done enough inner steps, break
                    if inner_step_count >= self.hparams.inner_steps:
                        break
        
        return {
            "total_loss": total_loss,
            "batch_count": batch_count,
            "batch_tokens": batch_tokens,
        }
    
    def outer_step(self, model, optimizer, scheduler=None):
        """
        Compute gradients between offloaded parameters and current parameters,
        then update with the outer optimizer.
        """
        if self.params_offloaded is None:
            raise ValueError("No offloaded parameters found. Run inner_step first.")
        
        actual_model = model.module if hasattr(model, 'module') else model
        model_params = list(actual_model.parameters())
        
        for param_offloaded, param in zip(self.params_offloaded, model_params):
            param_offloaded_on_device = param_offloaded.to(param.device)
            # Set gradients as the difference between original and updated parameters
            param.grad = param_offloaded_on_device - param.data
            
            # Only synchronize gradients when not using DeMo optimizer
            if self.world_size > 1 and self.config.outer_optimizer.lower() != "demo":
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)
            
            # Set parameter back to offloaded value for next inner step
            param.data = param_offloaded_on_device
        
        optimizer.step()
                
    def _get_offloaded_param(self, model):
        """Get a copy of current parameters and offload them to CPU"""
        actual_model = model.module if hasattr(model, 'module') else model
        return [param.data.detach().clone().to("cpu") for param in actual_model.parameters()]