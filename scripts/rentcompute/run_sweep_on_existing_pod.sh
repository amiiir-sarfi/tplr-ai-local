#!/usr/bin/env bash
set -euo pipefail

TARGET_POD_ID=""
WANDB_AGENT_ID=""
LOCAL_ENV_FILE_PATH="$HOME/tplr-ai-local/.env" # Default, can be overridden

# --- Helper Functions ---
print_usage() {
  echo "Usage: $0 -i <POD_ID> -a <WANDB_AGENT_ID> [-e <LOCAL_ENV_FILE_PATH>]"
  echo "  -i <POD_ID>               : The ID of the target Celium pod (from 'rentcompute list')."
  echo "  -a <WANDB_AGENT_ID>       : The W&B sweep/agent ID to run."
  echo "  -e <LOCAL_ENV_FILE_PATH>  : Optional path to the local .env file to be copied to the pod."
  echo "                              Defaults to '$LOCAL_ENV_FILE_PATH'."
  echo
  echo "If POD_ID is not provided, active instances will be listed."
}

# --- Argument Parsing ---
while getopts ":i:a:e:h" opt; do
  case ${opt} in
    i )
      TARGET_POD_ID=$OPTARG
      ;;
    a )
      WANDB_AGENT_ID=$OPTARG
      ;;
    e )
      LOCAL_ENV_FILE_PATH=$OPTARG
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

if [ -z "$WANDB_AGENT_ID" ]; then
  read -r -p "Please enter the W&B Agent ID to run: " WANDB_AGENT_ID
  if [ -z "$WANDB_AGENT_ID" ]; then
    echo "Error: W&B Agent ID is required."
    exit 1
  fi
fi

echo "--- Preparing to run job on Pod ID: $TARGET_POD_ID ---"
echo "W&B Agent ID: $WANDB_AGENT_ID"
echo "Local .env file: $LOCAL_ENV_FILE_PATH"

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

# Parse the line. Assuming fixed column order from 'rentcompute list'
# Name | ID | Host | User | Port | Status | GPU Type | GPU Count | Price ($/hr) | SSH Command
# $1   | $2 | $3   | $4   | $5   | $6     | $7       | $8        | $9           | $10 (approx)

# More robust parsing of fields:
export RENTCOMPUTE_POD_HOST=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
export RENTCOMPUTE_POD_USER=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $4); print $4}')
export RENTCOMPUTE_POD_PORT=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $5); print $5}')
SSH_COMMAND_FULL=$(echo "$pod_info_line" | awk -F ' *\\| *' '{gsub(/^[ \t]+|[ \t]+$/, "", $10); print $10}')

# Extract private key path from the SSH command string
# This uses grep with Perl-compatible regex (-P) to look for '-i ' and capture what follows until a space.
PRIVATE_KEY_FROM_CMD=$(echo "$SSH_COMMAND_FULL" | grep -oP -- '-i \K[^ ]+')
if [ -z "$PRIVATE_KEY_FROM_CMD" ]; then
  echo "Error: Could not parse SSH private key path from 'rentcompute list' output."
  echo "SSH Command found: $SSH_COMMAND_FULL"
  exit 1
fi
export RENTCOMPUTE_POD_KEY="$PRIVATE_KEY_FROM_CMD"
export RENTCOMPUTE_WANDB_AGENT="$WANDB_AGENT_ID"
export RENTCOMPUTE_LOCAL_ENV_PATH="$LOCAL_ENV_FILE_PATH" # This is passed to run_job_on_celium.sh

echo "--- Environment Variables Set ---"
echo "RENTCOMPUTE_POD_HOST: $RENTCOMPUTE_POD_HOST"
echo "RENTCOMPUTE_POD_USER: $RENTCOMPUTE_POD_USER"
echo "RENTCOMPUTE_POD_PORT: $RENTCOMPUTE_POD_PORT"
echo "RENTCOMPUTE_POD_KEY: $RENTCOMPUTE_POD_KEY"
echo "RENTCOMPUTE_WANDB_AGENT: $RENTCOMPUTE_WANDB_AGENT"
echo "RENTCOMPUTE_LOCAL_ENV_PATH: $RENTCOMPUTE_LOCAL_ENV_PATH"
echo "---------------------------------"

# Now, execute the main job script
# This script assumes run_job_on_celium.sh is in the same directory or in PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
JOB_SCRIPT_PATH="$SCRIPT_DIR/run_job_on_celium.sh"

if [ ! -f "$JOB_SCRIPT_PATH" ]; then
  echo "Error: Main job script '$JOB_SCRIPT_PATH' not found."
  exit 1
fi

echo "Executing '$JOB_SCRIPT_PATH'..."
bash "$JOB_SCRIPT_PATH"

echo "--- Manual job runner finished. ---"