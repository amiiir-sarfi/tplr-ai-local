import numpy as np
import torch
from torch.utils.data import DataLoader
from sharded_dataset import SharedShardedDataset            # ← your file name
from sharded_sampler import MinerSampler, EvalSampler

# ---------------------------------------------------------------------
# CONFIG
SHARDS_PATH   = "/home/shadeform/datasets/dclm_tokenized_llama2_cleaned"
SEQ_LEN       = 16
H             = 16          # steps_per_window
BATCH_SIZE    = 32          # global (all GPUs × grad-accum)
MICRO_BS      = 2           # per-GPU micro batch
VALIDATION_BS = 16          # ⩽ BATCH_SIZE

MINER_GPUS    = 8
VAL_GPUS      = 4
MINERS        = range(5)    # UIDs 0-4
WINDOWS       = range(7)    # 0..6
# ---------------------------------------------------------------------

# 1. Load shards once (rank-0 style) just to get dataset_len and data.
dataset = SharedShardedDataset(
    shards_path=SHARDS_PATH,
    sequence_length=SEQ_LEN,
    rank=0,
    world_size=1,
)
N = len(dataset)
print(f"{N:,} sequences loaded from shards")

def union_of_sets(list_of_sets):
    out = set()
    for s in list_of_sets:
        out |= s
    return out

# 2. Iterate over windows & miners ------------------------------------------------
for win in WINDOWS:
    for uid in MINERS:
        # ---- MINER TRAIN POOL ---------------------------------------------------
        miner_sets = []
        for rk in range(MINER_GPUS):
            ms = MinerSampler(
                dataset_len=N,
                uid=uid,
                window=win,
                steps_per_window=H,
                micro_bs=MICRO_BS,
                batch_size=BATCH_SIZE,
                rank=rk,
                world_size=MINER_GPUS,
            )
            miner_sets.append(set(ms))
        miner_pool = union_of_sets(miner_sets)

        # ---- VALIDATOR POOL -----------------------------------------------------
        val_sets = []
        for rk in range(VAL_GPUS):
            vs = EvalSampler(
                dataset_len=N,
                uid=uid,
                window=win,
                steps_per_window=H,
                micro_bs=MICRO_BS,
                batch_size=BATCH_SIZE,
                validation_bs=VALIDATION_BS,
                rank=rk,
                world_size=VAL_GPUS,
            )
            val_sets.append(set(vs))
        val_pool = union_of_sets(val_sets)

        # ---- MAKE SURE VALIDATOR DATA IS IN MINER DATA ---------------------------
        assert val_pool.issubset(miner_pool), (
            f"Validator found sample outside miner pool! "
            f"(uid={uid}, window={win})"
        )

    if win in (0, 3, 6):    # show a couple of examples
        ref_uid = 0
        print(f"\nWindow {win} – miner UID {ref_uid}")
        # first GPU (rank 0) samplers just to peek
        miner_s0 = MinerSampler(N, ref_uid, win, H, MICRO_BS, BATCH_SIZE, 0, MINER_GPUS)
        val_s0   = EvalSampler(N, ref_uid, win, H, MICRO_BS, BATCH_SIZE,
                               VALIDATION_BS, 0, VAL_GPUS)

        print("  miner rank-0 first 12 idx:", miner_s0._local[:12])
        print("  validator rank-0 first 12 idx:", val_s0._local[:12])

        # quick DataLoader sanity check
        loader = DataLoader(
            dataset,
            batch_size=MICRO_BS,
            sampler=miner_s0,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        x = next(iter(loader))
        print(f"  first DL batch shape: {tuple(x.shape)}")   # (micro_bs, seq_len)

print("\n✔ All validator pools are subsets of their miners.")
