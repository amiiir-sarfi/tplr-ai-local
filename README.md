<div align="center">

# τemplar: Incentivized Wide-Internet Training - Research Repository

</div>

<div align="center">
<pre>
___  _  _ _  _ | _  _
  | (/_| | ||_)|(_||
  |         |
</pre>
</div>

This repository contains the research codebase for [**τemplar**](https://github.com/tplr-ai/templar). This local version is primarily for development and experimentation.

## Local Setup (Ubuntu)

The following steps guide you through setting up the local environment.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/amiiir-sarfi/tplr-ai-local
    cd tplr-ai-local
    export TPLR_LOCAL_PATH=$(pwd)
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file by copying `.env.example`.
    ```bash
    cp .env.example .env
    ```
    Edit `.env` and ensure you add your `HF_TOKEN` from Hugging Face. This is crucial for pre-tokenizing data without encountering rate limits.
    Then, source the environment variables:
    ```bash
    source "${TPLR_LOCAL_PATH}/.env"
    # Alternative: export $(grep -v "^#" "${TPLR_LOCAL_PATH}/.env" | xargs)
    ```

3.  **Set Up System & Python Environment:**
    The [env_setup.sh](./env_setup.sh) script automates the installation of:
    *   Basic APT packages
    *   NVIDIA drivers
    *   CUDA toolkit & environment variables
    *   PM2 (process manager)
    *   UV (Python package installer and virtual environment manager)
    *   Hugging Face CLI login

    Execute the script and activate the virtual environment:
    ```bash
    ./env_setup.sh
    source "${TPLR_LOCAL_PATH}/.venv/bin/activate"
    ```

4.  **Pre-tokenize Data:**
    This step prepares the dataset for training (sharding and pretokenization). Choose between two datasets:
    
    **Option A: Default fineweb-edu-score-2 dataset** (using [pretokenize_data.py](./scripts/pretokenize_data.py)):
    ```bash
    # Define the output directory for sharded, tokenized data
    export DATA_SHARDED_DIR="$HOME/datasets/edu_fineweb_score2_10B_tokenized_llama2"

    python "${TPLR_LOCAL_PATH}/scripts/pretokenize_data.py"
    ```
    
    **Option B: DCLM dataset** (using [pretokenize_data_dclm.py](./scripts/pretokenize_data_dclm.py)):
    ```bash
    # Define the output directory for DCLM sharded, tokenized data
    export DATA_SHARDED_DIR="$HOME/datasets/dclm_tokenized_llama2_cleaned"

    python "${TPLR_LOCAL_PATH}/scripts/pretokenize_data_dclm.py"
    ```
    
    **IMPORTANT NOTES:** 
    - If `DATA_SHARDED_DIR` is modified here, ensure the `shards_path` argument in [neurons/torch_baseline_sharded.py](./neurons/torch_baseline_sharded.py) is updated accordingly. 
    - The script might raise an error upon completion even if all shards (e.g., 100) are successfully created; this can often be ignored if the output appears correct.
    - The DCLM pretokenization script uses multiprocessing and will automatically detect and use 75% of your available CPU cores for faster processing.

## Running Experiments

Ensure you are in the repository root and the virtual environment is activated:
```bash
cd "$TPLR_LOCAL_PATH"
source "${TPLR_LOCAL_PATH}/.venv/bin/activate"
```

### A. With WandB (W&B) Sweeps

**Important**: The hyperparameter files (hparams_file) within sweep configurations (e.g., in hparams/150M/sweeps/) assume this tplr-ai-local codebase is located at $HOME. If your TPLR_LOCAL_PATH is different, you must update the paths in these YAML configuration files.


Use the [run_sweep.sh](./run_sweep.sh) helper script to initiate a W&B sweep and immediately start an agent for the generated sweep ID. Examples are provided for a 150M parameter model.
```sh
cd $TPLR_LOCAL_PATH
source $TPLR_LOCAL_PATH/.venv/bin/activate
# 1. baseline Demo (no diloco) -- Loss = 2.83
bash ./run_sweep hparams/150M/sweeps/normal_demo/token3.7B.yaml

### DILOCO:
# 2. Demo + sign + no_momentum -- Loss=2.97
bash ./run_sweep hparams/150M/sweeps/diloco/demo_sign.yaml

# 3. Demo + no_sign + no_momentum -- Loss = 2.80
bash ./run_sweep hparams/150M/sweeps/diloco/demo_nosign.yaml

# 4. Demo + sign + momentum -- Loss > 4 -- This is under-experimented, first diving into (5) as its more likely to work.
bash ./run_sweep hparams/150M/sweeps/diloco/demo_momentum_sign.yaml

# 5. Demo + momentum -- Loss > 4 -- Performance is unexpectedly low given the "no_momentum_no_sign" variant (3) works well. Investigating.
bash ./run_sweep hparams/150M/sweeps/diloco/demo_momentum_nosign.yaml

# 6. baseline Diloco with Nesterov -- Loss = 2.78
bash ./run_sweep hparams/150M/sweeps/diloco/nesterov_baseline.yaml
```

### B. Directly with `torchrun` (Without W&B Sweeps)


Examples for running training scripts directly using `torchrun`.

1.  **Baseline Demo (No DiLoCo)**
    *   *Expected Loss: ~2.83*
    ```bash
    torchrun --nproc_per_node=4 neurons/torch_baseline_sharded.py \
    --project Demo150M_3.7B_Sharded \
    --token_budget=3774873600 \
    --sequence_length=2048 \
    --micro_batch_size=-1 \
    --hparams_file="${TPLR_LOCAL_PATH}/hparams/150M/150M_model_hparams.json" \
    --use_compile \
    --run_name=demo \
    --strategy=normal \
    --outer_optimizer=demo \
    --compression_decay=0.999 \
    --batch_size=32 \
    --outer_learning_rate=0.001 \
    --weight_decay=0.1 \
    --warmup_steps=750
    ```

2.  **DiLoCo: Demo + Full Gradient (No Sign Compression, No Momentum)**
    *   *This is currently the best-performing DemoDiLoCo configuration.*
    *   *Expected Loss: ~2.80*
    ```bash
    torchrun --nproc_per_node=8 neurons/torch_baseline_sharded.py \
    --project DemoDiloco150M_3.7B_Sharded \
    --token_budget=3774873600 \
    --sequence_length=2048 \
    --micro_batch_size=-1 \
    --hparams_file="${TPLR_LOCAL_PATH}/hparams/150M/150M_model_hparams.json" \
    --use_compile \
    --run_name=demodiloco \
    --strategy=diloco \
    --inner_optimizer=adamw \
    --outer_optimizer=demo \
    --compression_decay=0.999 \
    --warmup_steps=750 \
    --weight_decay=0.1 \
    --batch_size=32 \
    --inner_steps=10 \
    --inner_learning_rate=0.001 \
    --outer_learning_rate=0.9 \
    --outer_use_sign=0 \
    --outer_momentum=0.0
    ```

