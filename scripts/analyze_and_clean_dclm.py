import numpy as np
import os
import glob
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Analyze and reshard tokenized data")
    parser.add_argument("--data_dir", type=str, default="~/datasets/dclm_tokenized_llama2", help="Path to tokenized data directory")
    parser.add_argument("--target_token_size", type=int, default=100_000_000, help="Target number of tokens per shard")
    args = parser.parse_args()

    # Path to your tokenized data
    data_path = os.path.expandvars(os.path.expanduser(args.data_dir))
    
    # Create output directory
    output_dir = data_path + "_cleaned"
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load all .npy files
    npy_files = sorted(glob.glob(os.path.join(data_path, "train_*.npy")))
    print(f"Found {len(npy_files)} shard files")

    # Load and concatenate all data
    all_tokens = []
    for file in npy_files:
        tokens = np.load(file)
        all_tokens.append(tokens)
        print(f"Loaded {file}: {len(tokens):,} tokens")

    # Concatenate all tokens
    total_tokens = np.concatenate(all_tokens)
    print(f"\nTotal tokens: {len(total_tokens):,}")
    print(f"Total tokens (millions): {len(total_tokens)/1e6:.1f}M")

    # Calculate how many complete shards we can create
    num_complete_shards = int(len(total_tokens) // args.target_token_size)
    remaining_tokens = len(total_tokens) % args.target_token_size
    
    print(f"Target tokens per shard: {args.target_token_size:,}")
    print(f"Complete shards: {num_complete_shards:,}")
    print(f"Remaining tokens: {remaining_tokens:,}")

    # Reshard and save data
    for i in range(num_complete_shards):
        start_idx = i * args.target_token_size
        end_idx = (i + 1) * args.target_token_size
        shard_tokens = total_tokens[start_idx:end_idx]
        
        output_file = os.path.join(output_dir, f"train_{i:06d}.npy")
        np.save(output_file, shard_tokens)
        
        if i % 100 == 0 or i == num_complete_shards - 1:
            print(f"Saved shard {i+1}/{num_complete_shards}: {output_file}")
    
    # Save remaining tokens if any
    if remaining_tokens > 0:
        remaining_shard = total_tokens[-remaining_tokens:]
        output_file = os.path.join(output_dir, f"train_{num_complete_shards:06d}_partial.npy")
        np.save(output_file, remaining_shard)
        print(f"Saved partial shard: {output_file} ({remaining_tokens:,} tokens)")
    
    print(f"\nResharding complete! Saved {num_complete_shards} complete shards to {output_dir}")

if __name__ == "__main__":
    main()
