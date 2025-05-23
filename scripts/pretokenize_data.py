"""
Pre-tokenizes data from Hugging Face Hub (specifically HuggingFaceFW/fineweb-edu-score-2)
and saves it into shards.
Adapted from https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import math
import glob

# Global tokenizer for multiprocessing. Initialized in each worker.
_tokenizer_object = None

def initialize_worker_tokenizer(tokenizer_name_or_path):
    """Initializer function for each worker process in the pool."""
    global _tokenizer_object
    _tokenizer_object = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    _tokenizer_object.model_max_length = int(1e9) # suppress warning for long sequences
    if _tokenizer_object.eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer_name_or_path} must have an EOS token defined.")

def tokenize_document_entry(doc_entry):
    """
    Tokenizes a single document/text entry. Appends EOS token at the end.
    Returns a NumPy array of uint16 tokens or an empty array for invalid/empty input.
    """
    global _tokenizer_object
    text_content = doc_entry.get("text") # Use .get() for safety

    if not text_content or not isinstance(text_content, str): # Skip empty or non-string documents
        return np.array([], dtype=np.uint16)

    tokens = _tokenizer_object.encode(text_content, add_special_tokens=False)
    tokens.append(_tokenizer_object.eos_token_id)

    tokens_np_uint16 = np.array(tokens, dtype=np.uint16)

    if not ((0 <= tokens_np_uint16).all() and (tokens_np_uint16 < 2**16).all()):
        offending_indices = np.where(~((0 <= tokens_np_uint16) & (tokens_np_uint16 < 2**16)))[0]
        offending_token = tokens_np_uint16[offending_indices[0]]
        raise ValueError(
            f"Token ID {offending_token} (at index {offending_indices[0]} in document) is out of uint16 range. "
            f"Max vocab size for uint16 is 65535. Tokenizer vocab size: {_tokenizer_object.vocab_size}"
        )
    return tokens_np_uint16


def main(params):
    num_expected_shards = math.ceil(params.total_tokens_to_process / params.shard_size)
    
    if os.path.isdir(params.output_dir):
        # Count existing .npy files that match the shard naming pattern
        shard_pattern = os.path.join(params.output_dir, "train_*.npy")
        existing_shards = glob.glob(shard_pattern)
        num_existing_shards = 0
        for f_path in existing_shards:
            if os.path.isfile(f_path): # ensure it's a file
                 # Further check if the filename format is as expected (e.g., train_000000.npy)
                fname = os.path.basename(f_path)
                if fname.startswith("train_") and fname.endswith(".npy") and fname[6:-4].isdigit():
                    num_existing_shards +=1

        if num_existing_shards >= num_expected_shards:
            print(f"Output directory '{params.output_dir}' already exists and contains "
                  f"{num_existing_shards} (>= {num_expected_shards} expected) shard files.")
            print("Skipping pretokenization.")
            return # Exit the main function gracefully
        else:
            print(f"Output directory '{params.output_dir}' exists but has {num_existing_shards} shards, "
                  f"less than the expected {num_expected_shards}. Proceeding with tokenization.")

    os.makedirs(params.output_dir, exist_ok=True)

    print(f"Using tokenizer: {params.tokenizer_name}")
    # Quick check of tokenizer vocab size against uint16 limit before starting pool
    try:
        config = AutoConfig.from_pretrained(params.tokenizer_name)
        if config.vocab_size >= 2**16:
            print(f"Warning: Tokenizer vocab size ({config.vocab_size}) may exceed uint16 capacity.")
    except Exception as e:
        print(f"Could not pre-check tokenizer config: {e}")


    print(f"Loading dataset: {params.dataset_name} (config: {params.dataset_config_name or 'default'})")
    streamed_dataset = load_dataset(
        params.dataset_name,
        name=params.dataset_config_name, # Will be None for fineweb-edu-score-2 default
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    print(f"Shuffling dataset with seed {params.seed} and buffer size {params.shuffle_buffer_size}")
    streamed_dataset = streamed_dataset.shuffle(seed=params.seed, buffer_size=params.shuffle_buffer_size)

    tokenizer_processes = params.num_proc_tokenizer
    if tokenizer_processes == -1:
        tokenizer_processes = max(1, os.cpu_count() // 2)
    print(f"Starting tokenization with {tokenizer_processes} processes.")

    shard_idx = 0
    shard_token_buffer = np.empty((params.shard_size,), dtype=np.uint16)
    num_tokens_in_shard = 0
    total_tokens_processed_overall = 0

    estimated_shards= math.ceil(params.total_tokens_to_process / params.shard_size)
    print(f"Targeting {params.total_tokens_to_process / 1e9:.2f}B tokens.")
    print(f"Shard size: {params.shard_size / 1e6:.2f}M tokens. Estimated shards: {estimated_shards}")


    with mp.Pool(processes=tokenizer_processes,
                   initializer=initialize_worker_tokenizer,
                   initargs=(params.tokenizer_name,)) as token_pool:

        dataset_iterator = iter(streamed_dataset)
        
        with tqdm(total=params.total_tokens_to_process, unit="tokens", desc="Overall Token Progress") as progress_bar:
            for tokenized_doc_np in token_pool.imap(tokenize_document_entry, dataset_iterator, chunksize=params.processing_chunk_size):

                if total_tokens_processed_overall >= params.total_tokens_to_process:
                    break

                if len(tokenized_doc_np) == 0:
                    continue # Skip if document was empty or invalid

                doc_tokens_list = tokenized_doc_np.tolist()

                idx_in_doc = 0
                while idx_in_doc < len(doc_tokens_list):
                    if total_tokens_processed_overall >= params.total_tokens_to_process:
                        break

                    space_in_shard = params.shard_size - num_tokens_in_shard
                    tokens_left_in_doc = len(doc_tokens_list) - idx_in_doc
                    tokens_before_global_limit = params.total_tokens_to_process - total_tokens_processed_overall

                    num_to_take = min(space_in_shard, tokens_left_in_doc, tokens_before_global_limit)

                    if num_to_take == 0: # Should only happen if global limit is hit exactly or doc is empty
                        break

                    tokens_for_this_step = doc_tokens_list[idx_in_doc : idx_in_doc + num_to_take]

                    shard_token_buffer[num_tokens_in_shard : num_tokens_in_shard + num_to_take] = tokens_for_this_step
                    num_tokens_in_shard += num_to_take
                    total_tokens_processed_overall += num_to_take
                    progress_bar.update(num_to_take)
                    idx_in_doc += num_to_take

                    if num_tokens_in_shard == params.shard_size:
                        shard_filename = os.path.join(params.output_dir, f"train_{shard_idx:06d}.npy")
                        tqdm.write(f"Writing shard {shard_idx} ({num_tokens_in_shard / 1e6:.2f}M tokens) to {shard_filename}")
                        np.save(shard_filename, shard_token_buffer)
                        shard_idx += 1
                        num_tokens_in_shard = 0


                if total_tokens_processed_overall >= params.total_tokens_to_process:
                    break # Break from outer for-loop over documents

    # After the loop, if there are remaining tokens in the buffer, write them to a final shard
    if num_tokens_in_shard > 0 and total_tokens_processed_overall > 0: 
        final_shard_filename = os.path.join(params.output_dir, f"train_{shard_idx:06d}.npy")
        tqdm.write(f"Writing final shard {shard_idx} ({num_tokens_in_shard / 1e6:.2f}M tokens) to {final_shard_filename}")
        np.save(final_shard_filename, shard_token_buffer[:num_tokens_in_shard]) # Save only the filled part
        shard_idx += 1

    print(f"Completed tokenization. Total tokens processed and saved: {total_tokens_processed_overall:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-tokenize text data from Hugging Face Hub and save to shards.")

    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu-score-2",
                        help="Name of the dataset on Hugging Face Hub (default: 'HuggingFaceFW/fineweb-edu-score-2').")
    parser.add_argument("--dataset_config_name", type=str, default=None,
                        help="Configuration name for the dataset (e.g., 'sample-100BT' for original fineweb-edu). Default is None for fineweb-edu-score-2 which uses its default config.")
    parser.add_argument("--tokenizer_name", type=str, default="togethercomputer/LLaMA-2-7B-32K",
                        help="Name or path of the Hugging Face tokenizer (default: 'togethercomputer/LLaMA-2-7B-32K').")
    parser.add_argument("--output_dir", type=str, default="~/datasets/edu_fineweb_score2_10B_tokenized_llama2",
                        help="Directory to save the tokenized shards (default: '~/datasets/edu_fineweb_score2_10B_tokenized_llama2').")

    parser.add_argument("--shard_size", type=int, default=int(100e6),
                        help="Number of tokens per shard file (default: 100,000,000).")
    parser.add_argument("--total_tokens_to_process", type=int, default=int(10e9),
                        help="Total number of tokens to process and save (default: 10,000,000,000).")

    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for shuffling the dataset (default: 42).")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000,
                        help="Buffer size for shuffling the streamed dataset (default: 10000).")

    parser.add_argument("--num_proc_tokenizer", type=int, default=-1,
                        help="Worker processes for tokenization. -1 for os.cpu_count() // 2 (default: -1).")
    parser.add_argument("--processing_chunk_size", type=int, default=256,
                        help="Documents per worker before returning results (imap chunksize) (default: 256).")

    parsed_args = parser.parse_args()
    
    if isinstance(parsed_args.dataset_config_name, str) and parsed_args.dataset_config_name.lower() == "none":
        parsed_args.dataset_config_name = None
        
    parsed_args.output_dir = os.path.expanduser(parsed_args.output_dir)

    main(parsed_args)