#!/usr/bin/env bash
set -euo pipefail

TARGET_POD_ID=""
WANDB_AGENTS_STRING_INPUT="" # To store the space-separated input
LOCAL_ENV_FILE_PATH="$HOME/tplr-ai-local/.env" # Default, can be overridden
DATASET_TYPE="" # New parameter for dataset type

# --- Helper Functions ---
print_usage() {
  echo "Usage: $0 -i <POD_ID> -a <WANDB_AGENT_LIST_SPACE_SEPARATED> [-e <LOCAL_ENV_FILE_PATH>] [-d <DATASET_TYPE>]"
  echo "  -i <POD_ID>                             : The ID of the target Celium pod (from 'rentcompute list')."
  echo "  -a <WANDB_AGENT_LIST_SPACE_SEPARATED>   : Space-separated list of W&B sweep/agent IDs to run sequentially."
  echo "                                            If providing multiple, enclose in quotes: \"id1 id2 id3\"."
  echo "  -e <LOCAL_ENV_FILE_PATH>                : Optional path to the local .env file to be copied to the pod."
  echo "                                            Defaults to '$LOCAL_ENV_FILE_PATH'."
  echo "  -d <DATASET_TYPE>                       : Optional dataset type. Use 'dclm' for DCLM dataset,"
  echo "                                            otherwise defaults to fineweb-edu-score-2."
  echo
  echo "If POD_ID is not provided, active instances will be listed."
}

# --- Argument Parsing ---
while getopts ":i:a:e:d:h" opt; do
  case ${opt} in
    i )
      TARGET_POD_ID=$OPTARG
      ;;
    a )
      WANDB_AGENTS_STRING_INPUT=$OPTARG
      ;;
    e )
      LOCAL_ENV_FILE_PATH=$OPTARG
      ;;
    d )
      DATASET_TYPE=$OPTARG
      ;;
    h )
      print_usage
      exit 0
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      print_usage
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      print_usage
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

# --- Validate Required Arguments ---
if [ -z "$TARGET_POD_ID" ]; then
  echo "Error: Pod ID (-i) is required."
  echo "Listing active instances:"
  rentcompute list
  echo
  print_usage
  exit 1
fi

if [ -z "$WANDB_AGENTS_STRING_INPUT" ]; then
  read -r -p "Please enter the W&B Agent ID list (space-separated) to run: " WANDB_AGENTS_STRING_INPUT
  if [ -z "$WANDB_AGENTS_STRING_INPUT" ]; then
    echo "Error: W&B Agent ID list is required."
    exit 1
  fi
fi

# Convert space-separated string to comma-separated for the backend script
export RENTCOMPUTE_WANDB_AGENT_LIST=$(echo "$WANDB_AGENTS_STRING_INPUT" | tr ' ' ',')

echo "--- Preparing to run job on Pod ID: $TARGET_POD_ID ---"
echo "W&B Agent IDs (space-separated input): $WANDB_AGENTS_STRING_INPUT"
echo "W&B Agent IDs (comma-separated for script): $RENTCOMPUTE_WANDB_AGENT_LIST"
echo "Local .env file: $LOCAL_ENV_FILE_PATH"
echo "Dataset type: ${DATASET_TYPE:-default (fineweb-edu-score-2)}"

# --- Fetch Pod Details using rentcompute list ---
echo "Fetching instance details for Pod ID '$TARGET_POD_ID'..."
pod_info_line=$(rentcompute list | awk -F ' *\\| *' -v target_id="$TARGET_POD_ID" 'NR > 2 && $2 == target_id {print}')

if [ -z "$pod_info_line" ]; then
  echo "Error: Could not find Pod with ID '$TARGET_POD_ID' in 'rentcompute list' output."
  echo "Make sure the ID is correct and the instance is running."
  rentcompute list
  exit 1
fi

echo "Found pod info line: $pod_info_line"

export RENTCOMPUTE_POD_HOST=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
export RENTCOMPUTE_POD_USER=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}')
export RENTCOMPUTE_POD_PORT=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $5); print $5}')
SSH_COMMAND_FULL=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $10); print $10}')

PRIVATE_KEY_FROM_CMD=$(echo "$SSH_COMMAND_FULL" | grep -oP -- '-i \K[^ ]+')
if [ -z "$PRIVATE_KEY_FROM_CMD" ]; then
  echo "Error: Could not parse SSH private key path from 'rentcompute list' output."
  echo "SSH Command found: $SSH_COMMAND_FULL"
  exit 1
fi
export RENTCOMPUTE_POD_KEY="$PRIVATE_KEY_FROM_CMD"
# RENTCOMPUTE_WANDB_AGENT_LIST is already exported above
export RENTCOMPUTE_LOCAL_ENV_PATH="$LOCAL_ENV_FILE_PATH"
export RENTCOMPUTE_DATASET_TYPE="$DATASET_TYPE"

echo "--- Environment Variables Set ---"
echo "RENTCOMPUTE_POD_HOST: $RENTCOMPUTE_POD_HOST"
echo "RENTCOMPUTE_POD_USER: $RENTCOMPUTE_POD_USER"
echo "RENTCOMPUTE_POD_PORT: $RENTCOMPUTE_POD_PORT"
echo "RENTCOMPUTE_POD_KEY: $RENTCOMPUTE_POD_KEY"
echo "RENTCOMPUTE_WANDB_AGENT_LIST (for script): $RENTCOMPUTE_WANDB_AGENT_LIST"
echo "RENTCOMPUTE_LOCAL_ENV_PATH: $RENTCOMPUTE_LOCAL_ENV_PATH"
echo "RENTCOMPUTE_DATASET_TYPE: $RENTCOMPUTE_DATASET_TYPE"
echo "---------------------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
JOB_SCRIPT_PATH="$SCRIPT_DIR/run_job_on_celium.sh"

if [ ! -f "$JOB_SCRIPT_PATH" ]; then
  echo "Error: Main job script '$JOB_SCRIPT_PATH' not found."
  exit 1
fi

echo "Executing '$JOB_SCRIPT_PATH'..."
bash "$JOB_SCRIPT_PATH"

echo "--- Manual job runner finished. ---"