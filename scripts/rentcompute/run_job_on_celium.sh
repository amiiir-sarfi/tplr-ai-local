#!/usr/bin/env bash
set -euo pipefail

echo "--- Celium Job Runner & Provisioner ---"
echo "Remote Pod Host: $RENTCOMPUTE_POD_HOST"
echo "Remote Pod User: $RENTCOMPUTE_POD_USER"
echo "Remote Pod Port: $RENTCOMPUTE_POD_PORT"

PRIVATE_KEY_PATH="${RENTCOMPUTE_POD_KEY%.pub}" # Assumes .pub extension
REMOTE_SETUP_SCRIPT_NAME="celium_env_setup.sh"
REMOTE_SETUP_SCRIPT_LOCAL_PATH="./${REMOTE_SETUP_SCRIPT_NAME}"
REMOTE_SCRIPT_DEST_ON_SERVER="~/${REMOTE_SETUP_SCRIPT_NAME}"
REMOTE_ENV_FILE_DEST="~/.env"
LOG_FILE_REMOTE="~/job_runner.log"

LOCAL_DATASET_TAR_PATH="~/dataset.tar"
EXPANDED_LOCAL_DATASET_TAR_PATH=$(eval echo "$LOCAL_DATASET_TAR_PATH")

if [ -z "$RENTCOMPUTE_POD_HOST" ]; then
  echo "Error: RENTCOMPUTE_POD_HOST is not set. Cannot proceed."
  exit 1
fi

echo "Ensuring local setup script '$REMOTE_SETUP_SCRIPT_LOCAL_PATH' is executable..."
chmod +x "$REMOTE_SETUP_SCRIPT_LOCAL_PATH"

echo "Copying setup script '$REMOTE_SETUP_SCRIPT_LOCAL_PATH' to '$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST:$REMOTE_SCRIPT_DEST_ON_SERVER'..."
scp -P "$RENTCOMPUTE_POD_PORT" -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$REMOTE_SETUP_SCRIPT_LOCAL_PATH" \
  "$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST:$REMOTE_SCRIPT_DEST_ON_SERVER"

if [ -n "${RENTCOMPUTE_LOCAL_ENV_PATH:-}" ] && [ -f "$RENTCOMPUTE_LOCAL_ENV_PATH" ]; then
  echo "Copying local .env file from '$RENTCOMPUTE_LOCAL_ENV_PATH' to '$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST:$REMOTE_ENV_FILE_DEST'..."
  scp -P "$RENTCOMPUTE_POD_PORT" -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$RENTCOMPUTE_LOCAL_ENV_PATH" \
    "$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST:$REMOTE_ENV_FILE_DEST"
else
  echo "Warning: RENTCOMPUTE_LOCAL_ENV_PATH ('${RENTCOMPUTE_LOCAL_ENV_PATH:-}') not set or file not found. .env file not copied."
fi


JOB_EXECUTION_SCRIPT_CONTENT='#!/usr/bin/env bash
set -e
echo "--- Background Job Script Started ---"
echo "Timestamp: $(date)"
echo "Attempting to source environment variables from $REMOTE_ENV_FILE_DEST..."
export $(grep -v "^#" $HOME/.env | xargs) || { echo "Failed to source environment variables from $REMOTE_ENV_FILE_DEST. Exiting."; exit 1; }
echo "Changing to ~/tplr-ai-local directory..."
cd ~/tplr-ai-local || { echo "Failed to cd to ~/tplr-ai-local. This directory should have been cloned by celium_env_setup.sh."; exit 1; }

echo "Ensuring Python venv is active (explicitly)..."
VENV_PATH="$HOME/tplr-ai-local/.venv" # Define VENV_PATH for clarity
if [ -d "${VENV_PATH}/bin" ]; then
  source "${VENV_PATH}/bin/activate"
  echo "Activated .venv."
else
  echo "Venv ${VENV_PATH}/bin not found in ~/tplr-ai-local. Environment setup might be incomplete."
fi

echo "Current PATH after venv attempt: ${PATH}"
WANDB_EXECUTABLE="${VENV_PATH}/bin/wandb"
echo "Checking for wandb executable at: ${WANDB_EXECUTABLE}"
if ! command -v "${WANDB_EXECUTABLE}" &> /dev/null; then
    echo "Error: wandb executable not found at ${WANDB_EXECUTABLE} even after attempting to activate venv."
    echo "Please check celium_env_setup.sh and venv integrity."
    exit 1
fi
echo "wandb executable found."

AGENT_LIST_STR="${RENTCOMPUTE_WANDB_AGENT_LIST}" # This will be expanded by the shell running this script (run_job_background.sh)

if [ -n "${AGENT_LIST_STR}" ]; then
  echo "Found W\&B agent list: '\''${AGENT_LIST_STR}'\''" # Use single quotes to show the exact string
  OLD_IFS="${IFS}"
  IFS=","
  AGENT_COUNT=0
  # Calculate total agents more robustly
  TOTAL_AGENTS=$(echo "${AGENT_LIST_STR}" | tr "," "\n" | wc -l)
  CURRENT_AGENT_NUM=0

  for AGENT_ID_TO_RUN in ${AGENT_LIST_STR}; do # relies on IFS splitting
    CURRENT_AGENT_NUM=$((CURRENT_AGENT_NUM + 1))
    IFS="${OLD_IFS}" # Restore IFS for commands within the loop

    echo "Starting W\&B agent (${CURRENT_AGENT_NUM}/${TOTAL_AGENTS}): '\''${AGENT_ID_TO_RUN}'\''"
    "${WANDB_EXECUTABLE}" agent "${AGENT_ID_TO_RUN}"
    AGENT_EXIT_CODE=$?
    echo "W\&B agent '\''${AGENT_ID_TO_RUN}'\'' finished with exit code: ${AGENT_EXIT_CODE}."

    if [ ${AGENT_EXIT_CODE} -ne 0 ]; then
       echo "Error: Agent '\''${AGENT_ID_TO_RUN}'\'' failed with exit code ${AGENT_EXIT_CODE}. Stopping further agents in the list."
       # Optionally, exit the script if one agent fails:
       # exit ${AGENT_EXIT_CODE}
    fi
    AGENT_COUNT=$((AGENT_COUNT + 1))
    IFS="," # Re-set IFS for the loop
  done
  IFS="${OLD_IFS}"
  echo "All ${AGENT_COUNT} W\&B agents in the list have been processed."
else
  echo "No RENTCOMPUTE_WANDB_AGENT_LIST provided. No W\&B agents will be run by this script."
fi
'

REMOTE_COMMAND_SEQUENCE="set -e; echo '--- Starting Remote Execution (Main SSH Session) ---'; "
REMOTE_COMMAND_SEQUENCE+="echo 'Running environment setup script: $REMOTE_SCRIPT_DEST_ON_SERVER'; "
# Ensure celium_env_setup.sh has correct permissions and is executed correctly
REMOTE_COMMAND_SEQUENCE+="chmod +x ${REMOTE_SCRIPT_DEST_ON_SERVER}; bash ${REMOTE_SCRIPT_DEST_ON_SERVER}; "


if [ -f "$EXPANDED_LOCAL_DATASET_TAR_PATH" ]; then
  echo "Local dataset tarball '$EXPANDED_LOCAL_DATASET_TAR_PATH' found. Copying to remote..."
  scp -P "$RENTCOMPUTE_POD_PORT" -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$EXPANDED_LOCAL_DATASET_TAR_PATH" \
    "$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST:~/"

  REMOTE_COMMAND_SEQUENCE+="echo 'Extracting ~/dataset.tar into ~/ ...'; "
  REMOTE_COMMAND_SEQUENCE+="tar xf ~/dataset.tar -C ~/; "
else
  echo "Local dataset tarball '$EXPANDED_LOCAL_DATASET_TAR_PATH' not found. Will attempt remote tokenization."
  
  # Determine which pretokenization script to use
  if [ "${RENTCOMPUTE_DATASET_TYPE:-}" = "dclm" ]; then
    PRETOKENIZE_SCRIPT="scripts/pretokenize_data_dclm.py"
    echo "Using DCLM dataset - will run $PRETOKENIZE_SCRIPT"
  else
    PRETOKENIZE_SCRIPT="scripts/pretokenize_data.py"
    echo "Using default dataset (fineweb-edu-score-2) - will run $PRETOKENIZE_SCRIPT"
  fi
  
  PRETOKENIZE_CMD="echo 'Local dataset.tar not found. Attempting to run $PRETOKENIZE_SCRIPT on remote server.'; "
  PRETOKENIZE_CMD+="echo 'Running $PRETOKENIZE_SCRIPT script using venv python with retries...'; "
  
  PRETOKENIZE_CMD+="MAX_RETRIES=5; "
  PRETOKENIZE_CMD+="RETRY_COUNT=0; "
  PRETOKENIZE_CMD+="SUCCESS=false; "
  PRETOKENIZE_CMD+="echo 'Max retries for pretokenization: \${MAX_RETRIES}'; " 
  PRETOKENIZE_CMD+="while [ \"\${RETRY_COUNT}\" -lt \"\${MAX_RETRIES}\" ]; do "
  PRETOKENIZE_CMD+="  echo \"Attempt \$((RETRY_COUNT + 1))/\${MAX_RETRIES} to run $PRETOKENIZE_SCRIPT...\"; "
  # Use full path to python in venv for pretokenize script
  PRETOKENIZE_CMD+="  if (cd ~/tplr-ai-local/ && \$HOME/tplr-ai-local/.venv/bin/python $PRETOKENIZE_SCRIPT && cd ~); then "
  PRETOKENIZE_CMD+="    echo 'Pretokenize script successful (or skipped due to existing data) on attempt \$((RETRY_COUNT + 1)).'; "
  PRETOKENIZE_CMD+="    SUCCESS=true; "
  PRETOKENIZE_CMD+="    break; " 
  PRETOKENIZE_CMD+="  else "
  PRETOKENIZE_CMD+="    SCRIPT_EXIT_CODE=\$?; " 
  PRETOKENIZE_CMD+="    echo 'Pretokenize script failed on attempt \$((RETRY_COUNT + 1)) with exit code \${SCRIPT_EXIT_CODE}.'; "
  PRETOKENIZE_CMD+="    RETRY_COUNT=\$((RETRY_COUNT + 1)); "
  PRETOKENIZE_CMD+="    if [ \"\${RETRY_COUNT}\" -lt \"\${MAX_RETRIES}\" ]; then "
  PRETOKENIZE_CMD+="      echo 'Retrying in 15 seconds...'; "
  PRETOKENIZE_CMD+="      sleep 15; "
  PRETOKENIZE_CMD+="    fi; "
  PRETOKENIZE_CMD+="  fi; " 
  PRETOKENIZE_CMD+="done; " 
  
  PRETOKENIZE_CMD+="if [ \"\${SUCCESS}\" = false ]; then "
  PRETOKENIZE_CMD+="  echo 'Error: Pretokenize script failed after \${MAX_RETRIES} attempts.'; "
  PRETOKENIZE_CMD+="  exit 1; " 
  PRETOKENIZE_CMD+="else "
  PRETOKENIZE_CMD+="  echo 'Pretokenization step completed.'; " 
  
  # Add cleanup step for DCLM dataset
  if [ "${RENTCOMPUTE_DATASET_TYPE:-}" = "dclm" ]; then
    PRETOKENIZE_CMD+="  echo 'Running DCLM data cleanup and resharding...'; "
    PRETOKENIZE_CMD+="  if (cd ~/tplr-ai-local/ && \$HOME/tplr-ai-local/.venv/bin/python scripts/analyze_and_clean_dclm.py && cd ~); then "
    PRETOKENIZE_CMD+="    echo 'DCLM data cleanup completed successfully.'; "
    PRETOKENIZE_CMD+="  else "
    PRETOKENIZE_CMD+="    echo 'Warning: DCLM data cleanup failed, but continuing...'; "
    PRETOKENIZE_CMD+="  fi; "
  fi
  
  PRETOKENIZE_CMD+="fi; "
  
  REMOTE_COMMAND_SEQUENCE+="$PRETOKENIZE_CMD"
fi

# Check if RENTCOMPUTE_WANDB_AGENT_LIST is set and non-empty in *this* script's environment
if [ -n "${RENTCOMPUTE_WANDB_AGENT_LIST:-}" ]; then
  echo "W&B Agent ID list ('$RENTCOMPUTE_WANDB_AGENT_LIST') provided by parent. Will create background script to run agents sequentially."
  REMOTE_COMMAND_SEQUENCE+="echo '--- Post-setup: Launching W\&B Agents Sequentially in Background ---'; "
  ESCAPED_JOB_EXECUTION_SCRIPT_CONTENT=$(echo "$JOB_EXECUTION_SCRIPT_CONTENT" | sed "s/'/'\\\\''/g")

  REMOTE_COMMAND_SEQUENCE+="printf '%s' '$ESCAPED_JOB_EXECUTION_SCRIPT_CONTENT' > ~/run_job_background.sh; "
  REMOTE_COMMAND_SEQUENCE+="chmod +x ~/run_job_background.sh; "
  REMOTE_COMMAND_SEQUENCE+="echo 'Launching ~/run_job_background.sh with nohup... Output will be in ${LOG_FILE_REMOTE}'; "
  # Ensure RENTCOMPUTE_WANDB_AGENT_LIST is available to nohup
  # nohup inherits the environment by default. The issue is if it's not exported or changed.
  # To be explicit:
REMOTE_COMMAND_SEQUENCE+="export REMOTE_ENV_FILE_DEST=\"${REMOTE_ENV_FILE_DEST}\"; export RENTCOMPUTE_WANDB_AGENT_LIST=\"${RENTCOMPUTE_WANDB_AGENT_LIST}\"; export RENTCOMPUTE_DATASET_TYPE=\"${RENTCOMPUTE_DATASET_TYPE:-}\"; nohup ~/run_job_background.sh > ${LOG_FILE_REMOTE} 2>&1 & "

else
  echo "No RENTCOMPUTE_WANDB_AGENT_LIST set by parent. Server will remain running after initial setup."
  REMOTE_COMMAND_SEQUENCE+="echo 'Environment and data setup complete. No background job (W\&B agent list) to run.' ;"
fi

REMOTE_COMMAND_SEQUENCE+="echo '--- Remote Execution (Main SSH Session) Finished ---'; "

echo "Executing the following command sequence on remote host '$RENTCOMPUTE_POD_HOST':"
echo "----------------------------------------------------"

# Execute the command sequence
ssh -p "$RENTCOMPUTE_POD_PORT" -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST" \
  "$REMOTE_COMMAND_SEQUENCE"

echo "--- Remote provisioning/job script wrapper finished. ---"
if [ -n "${RENTCOMPUTE_WANDB_AGENT_LIST:-}" ]; then
  echo "The W&B agents have been launched sequentially in the background on the remote server."
  echo "Monitor their progress by SSHing into the server and checking the log file: tail -f ${LOG_FILE_REMOTE}"
  echo "You can also check the raw script written to the remote: ssh $RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST -p $RENTCOMPUTE_POD_PORT -i $PRIVATE_KEY_PATH cat ~/run_job_background.sh"
fi
echo "Your local rentcompute command can now exit."