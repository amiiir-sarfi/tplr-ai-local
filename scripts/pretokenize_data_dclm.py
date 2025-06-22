import tplr
from transformers import AutoTokenizer
import torch
import asyncio
import numpy as np
import os
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager, Process
import time
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Pretokenize DCLM dataset")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=6, help="Batch size")
    parser.add_argument("--max_iterations", type=int, default=3000, help="Maximum iterations")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--pages_per_window", type=int, default=6, help="Pages per window")
    parser.add_argument("--tokenizer_name", type=str, default="togethercomputer/LLaMA-2-7B-32K", help="Tokenizer name")
    parser.add_argument("--seed", type=str, default="adam_baseline", help="Random seed")
    parser.add_argument("--shard_size", type=int, default=int(100e6), help="Shard size in tokens")
    parser.add_argument("--output_dir", type=str, default="~/datasets/dclm_tokenized_llama2_validation", help="Output directory")
    parser.add_argument("--validation_offset", type=int, default=-1, help="Validation offset")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes (auto-detect if not specified)")
    return parser.parse_args()

args = parse_args()

seq_len = args.seq_len
batch_size = args.batch_size
max_iterations = args.max_iterations
workers = args.workers
pages_per_window = args.pages_per_window
tokenizer_name = args.tokenizer_name
seed = args.seed
shard_size = args.shard_size
output_dir = args.output_dir
validation_offset = args.validation_offset

validation_offset = max(0, validation_offset)
training = not (validation_offset > 0)
if not training:
    print(f"Warning: Validation offset is HARDCODED to {validation_offset} for reproducability.")
    validation_offset = 5500 # Hardcoded to ensure reproducability
    max_iterations = 17
shard_prefix = "train_" if training else "validation_"
# Auto-detect optimal number of processes

cpu_count = mp.cpu_count()
if training:
    num_processes = max(1, int(cpu_count * 0.75))
else:
    num_processes = 1  # For validation, use a single process
print(f"Detected {cpu_count} CPU cores, using {num_processes} processes for {shard_prefix} data")

async def process_worker_async(process_id, start_window, end_window, output_path, shared_shard_counter, shard_counter_lock, progress_queue):
    """Async worker function for a single process"""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, verbose=False, clean_up_tokenization_spaces=True
    )
    print(f"Process {process_id}: vocab size = {tokenizer.vocab_size}")
    sys.stdout.flush()
    
    shard_buffer = np.empty((shard_size,), dtype=np.uint16)
    tokens_in_buffer = 0
    total_tokens_processed = 0
    
    for window in range(start_window, end_window):
        window_tokens = 0
        
        for worker in range(workers):
            step_window = window * workers + worker
            
            try:
                pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                    offset=step_window * pages_per_window + validation_offset,
                    n_pages=pages_per_window,
                    seed=seed,
                )

                loader = await tplr.r2_dataset.R2DatasetLoader.create(
                    batch_size=batch_size,
                    sequence_length=seq_len,
                    pages_info=pages,
                    tokenizer=tokenizer,
                )
                
                for i, batch in enumerate(loader):
                    input_ids = torch.tensor(batch, dtype=torch.long)
                    flat_tokens = input_ids.flatten().numpy().astype(np.uint16)
                    total_tokens_processed += len(flat_tokens)
                    window_tokens += len(flat_tokens)
                    
                    # Add tokens to shard buffer
                    tokens_to_add = len(flat_tokens)
                    idx_in_batch = 0
                    
                    while idx_in_batch < tokens_to_add:
                        space_in_shard = shard_size - tokens_in_buffer
                        tokens_left_in_batch = tokens_to_add - idx_in_batch
                        num_to_take = min(space_in_shard, tokens_left_in_batch)
                        
                        if num_to_take > 0:
                            batch_tokens = flat_tokens[idx_in_batch:idx_in_batch + num_to_take]
                            shard_buffer[tokens_in_buffer:tokens_in_buffer + num_to_take] = batch_tokens
                            tokens_in_buffer += num_to_take
                            idx_in_batch += num_to_take
                        
                        # Save shard when full
                        if tokens_in_buffer == shard_size:
                            with shard_counter_lock:
                                shard_idx = shared_shard_counter.value
                                shared_shard_counter.value += 1

                            shard_filename = os.path.join(output_path, f"{shard_prefix}{shard_idx:06d}.npy")
                            np.save(shard_filename, shard_buffer.copy())
                            
                            progress_queue.put(f"Process {process_id}: Saved shard {shard_idx} with {shard_size/1e6:.1f}M tokens")
                            tokens_in_buffer = 0
                
                # Log progress for process 0 after each loader
                if process_id == 0:
                    progress_queue.put(f"[Process 0]: Total tokens processed so far: {total_tokens_processed/1e6:.1f}M at {window=}")
            
            except Exception as e:
                progress_queue.put(f"Process {process_id}: Error processing window {window}, worker {worker}: {e}")
                continue
        
        # Update progress
        progress_queue.put(1)
    
    # Save any remaining tokens
    if tokens_in_buffer > 0:
        with shard_counter_lock:
            shard_idx = shared_shard_counter.value
            shared_shard_counter.value += 1

        final_shard_filename = os.path.join(output_path, f"{shard_prefix}{shard_idx:06d}.npy")
        np.save(final_shard_filename, shard_buffer[:tokens_in_buffer])
        progress_queue.put(f"Process {process_id}: Saved final shard {shard_idx} with {tokens_in_buffer/1e6:.1f}M tokens")
    
    progress_queue.put(f"Process {process_id}: Completed! Total tokens processed: {total_tokens_processed/1e9:.2f}B")

def process_worker(process_id, start_window, end_window, output_path, shared_shard_counter, shard_counter_lock, progress_queue):
    """Wrapper to run async worker in a process"""
    asyncio.run(process_worker_async(process_id, start_window, end_window, output_path, shared_shard_counter, shard_counter_lock, progress_queue))

async def main():
    output_path = os.path.expanduser(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    # Create shared state
    manager = Manager()
    shared_shard_counter = mp.Value('i', 0)
    shard_counter_lock = mp.Lock()
    progress_queue = manager.Queue()
    
    # Split work among processes
    windows_per_process = max_iterations // num_processes
    processes = []
    
    print(f"Starting {num_processes} processes to process {max_iterations} windows")
    
    # Start worker processes
    for process_id in range(num_processes):
        start_window = process_id * windows_per_process
        end_window = start_window + windows_per_process
        if process_id == num_processes - 1:  # Last process handles remainder
            end_window = max_iterations
            
        p = Process(
            target=process_worker,
            args=(process_id, start_window, end_window, output_path, shared_shard_counter, shard_counter_lock, progress_queue)
        )
        p.start()
        processes.append(p)
        print(f"Started process {process_id} handling windows {start_window}-{end_window-1}")
    
    # Monitor progress
    completed_windows = 0
    
    with tqdm(total=max_iterations, desc="Processing windows") as pbar:
        while completed_windows < max_iterations:
            try:
                while not progress_queue.empty():
                    message = progress_queue.get_nowait()
                    if isinstance(message, str):
                        print(message)
                        sys.stdout.flush()
                    else:
                        completed_windows += message
                        pbar.update(message)
                    
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in progress monitoring: {e}")
                break
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print(f"All processes completed!")
    print(f"Total shards created: {shared_shard_counter.value}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asyncio.run(main())
    print("Done!")