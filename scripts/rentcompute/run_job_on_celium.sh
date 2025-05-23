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

JOB_EXECUTION_SCRIPT_CONTENT="#!/usr/bin/env bash
set -e 
echo '--- Background Job Script Started ---'
echo 'Timestamp: \$(date)'
echo 'Attempting to source environment variables from $REMOTE_ENV_FILE_DEST...'
if [ -f \"$REMOTE_ENV_FILE_DEST\" ]; then
  set -o allexport 
  source \"$REMOTE_ENV_FILE_DEST\"
  set +o allexport
  echo 'Sourced variables from .env file.'
else
  echo 'Remote .env file ($REMOTE_ENV_FILE_DEST) not found, skipping sourcing.'
fi
echo 'Changing to ~/tplr-ai-local directory...'
cd ~/tplr-ai-local || { echo 'Failed to cd to ~/tplr-ai-local. This directory should have been cloned by celium_env_setup.sh.'; exit 1; }
echo 'Ensuring Python venv is active (explicitly)...'
if [ -d .venv/bin ]; then
  source .venv/bin/activate
  echo 'Activated .venv.'
else
  echo 'Venv .venv/bin not found in ~/tplr-ai-local. Environment setup might be incomplete.'
fi
echo 'Current PATH: \$PATH'
echo 'Which wandb: $(which wandb || echo "wandb not found in PATH")'
echo 'Starting wandb agent: $RENTCOMPUTE_WANDB_AGENT'
wandb agent '$RENTCOMPUTE_WANDB_AGENT' 
AGENT_EXIT_CODE=\$?
echo \"W\&B agent finished with exit code: \$AGENT_EXIT_CODE.\"
"

REMOTE_COMMAND_SEQUENCE="set -e; echo '--- Starting Remote Execution (Main SSH Session) ---'; "
REMOTE_COMMAND_SEQUENCE+="echo 'Running environment setup script: $REMOTE_SCRIPT_DEST_ON_SERVER'; "
REMOTE_COMMAND_SEQUENCE+="bash ${REMOTE_SCRIPT_DEST_ON_SERVER}; "

if [ -f "$EXPANDED_LOCAL_DATASET_TAR_PATH" ]; then
  echo "Local dataset tarball '$EXPANDED_LOCAL_DATASET_TAR_PATH' found. Copying to remote..."
  scp -P "$RENTCOMPUTE_POD_PORT" -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$EXPANDED_LOCAL_DATASET_TAR_PATH" \
    "$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST:~/"

  REMOTE_COMMAND_SEQUENCE+="echo 'Extracting ~/dataset.tar into ~/ ...'; "
  REMOTE_COMMAND_SEQUENCE+="tar xf ~/dataset.tar -C ~/; "
else
  echo "Local dataset tarball '$EXPANDED_LOCAL_DATASET_TAR_PATH' not found. Will attempt remote tokenization."
  
  # Construct the command string for remote execution carefully
  PRETOKENIZE_CMD="echo 'Local dataset.tar not found. Attempting to run pretokenize_data.py on remote server.'; "
  PRETOKENIZE_CMD+="echo 'Running pretokenize_data.py script using venv python with retries...'; "
  
  # Retry logic for pretokenization
  PRETOKENIZE_CMD+="MAX_RETRIES=5; "
  PRETOKENIZE_CMD+="RETRY_COUNT=0; "
  PRETOKENIZE_CMD+="SUCCESS=false; "
  PRETOKENIZE_CMD+="echo 'Max retries for pretokenization: \$MAX_RETRIES'; " # Log max_retries
  PRETOKENIZE_CMD+="while [ \"\$RETRY_COUNT\" -lt \"\$MAX_RETRIES\" ]; do "
  PRETOKENIZE_CMD+="  echo \"Attempt \$((RETRY_COUNT + 1))/\$MAX_RETRIES to run pretokenize_data.py...\"; "
  PRETOKENIZE_CMD+="  if (cd ~/tplr-ai-local/ && ~/tplr-ai-local/.venv/bin/python scripts/pretokenize_data.py && cd ~); then "
  PRETOKENIZE_CMD+="    echo 'Pretokenize script successful (or skipped due to existing data) on attempt \$((RETRY_COUNT + 1)).'; "
  PRETOKENIZE_CMD+="    SUCCESS=true; "
  PRETOKENIZE_CMD+="    break; " # Exit the while loop
  PRETOKENIZE_CMD+="  else "
  PRETOKENIZE_CMD+="    SCRIPT_EXIT_CODE=\$?; " # Capture exit code of the subshell
  PRETOKENIZE_CMD+="    echo 'Pretokenize script failed on attempt \$((RETRY_COUNT + 1)) with exit code \$SCRIPT_EXIT_CODE.'; "
  PRETOKENIZE_CMD+="    RETRY_COUNT=\$((RETRY_COUNT + 1)); "
  PRETOKENIZE_CMD+="    if [ \"\$RETRY_COUNT\" -lt \"\$MAX_RETRIES\" ]; then "
  PRETOKENIZE_CMD+="      echo 'Retrying in 15 seconds...'; "
  PRETOKENIZE_CMD+="      sleep 15; "
  PRETOKENIZE_CMD+="    fi; "
  PRETOKENIZE_CMD+="  fi; " # End of if/else for command execution
  PRETOKENIZE_CMD+="done; " # End of while loop
  
  PRETOKENIZE_CMD+="if [ \"\$SUCCESS\" = false ]; then "
  PRETOKENIZE_CMD+="  echo 'Error: Pretokenize script failed after \$MAX_RETRIES attempts.'; "
  PRETOKENIZE_CMD+="  exit 1; " # This will terminate the remote command sequence due to 'set -e' at the start of REMOTE_COMMAND_SEQUENCE
  PRETOKENIZE_CMD+="else "
  PRETOKENIZE_CMD+="  echo 'Pretokenization step completed.'; " # Covers success or skip
  PRETOKENIZE_CMD+="fi; "
  
  REMOTE_COMMAND_SEQUENCE+="$PRETOKENIZE_CMD"
fi

if [ -n "${RENTCOMPUTE_WANDB_AGENT:-}" ]; then
  echo "W&B Agent ID ('$RENTCOMPUTE_WANDB_AGENT') provided. Will run agent in background."
  REMOTE_COMMAND_SEQUENCE+="echo '--- Post-setup: Launching W\&B Agent in Background ---'; "
  REMOTE_COMMAND_SEQUENCE+="printf '%s' \"${JOB_EXECUTION_SCRIPT_CONTENT}\" > ~/run_job_background.sh; "
  REMOTE_COMMAND_SEQUENCE+="chmod +x ~/run_job_background.sh; "
  REMOTE_COMMAND_SEQUENCE+="echo 'Launching ~/run_job_background.sh with nohup... Output will be in ${LOG_FILE_REMOTE}'; "
  REMOTE_COMMAND_SEQUENCE+="nohup ~/run_job_background.sh > ${LOG_FILE_REMOTE} 2>&1 & "
else
  echo "No RENTCOMPUTE_WANDB_AGENT set. Server will remain running after initial setup."
  REMOTE_COMMAND_SEQUENCE+="echo 'Environment and data setup complete. No background job (W\&B agent) to run.' ;"
fi

REMOTE_COMMAND_SEQUENCE+="echo '--- Remote Execution (Main SSH Session) Finished ---'; "

echo "Executing the following command sequence on remote host '$RENTCOMPUTE_POD_HOST':"
echo "----------------------------------------------------"
printf "%s\n" "$REMOTE_COMMAND_SEQUENCE" 
echo "----------------------------------------------------"

ssh -p "$RENTCOMPUTE_POD_PORT" -i "$PRIVATE_KEY_PATH" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$RENTCOMPUTE_POD_USER@$RENTCOMPUTE_POD_HOST" \
  "$REMOTE_COMMAND_SEQUENCE"

echo "--- Remote provisioning/job script wrapper finished. ---"
if [ -n "${RENTCOMPUTE_WANDB_AGENT:-}" ]; then
  echo "The W&B agent has been launched in the background on the remote server."
  echo "Monitor its progress by SSHing into the server and checking the log file: tail -f ${LOG_FILE_REMOTE}"
fi
echo "Your local rentcompute command can now exit."