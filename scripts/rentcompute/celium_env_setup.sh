#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

echo "--- Celium Node Environment Setup Script ---"

# Ensure scripts run non-interactively
export DEBIAN_FRONTEND=noninteractive

# Determine sudo prefix if needed
if [[ "$(id -u)" -ne 0 ]] && command -v sudo &>/dev/null; then
  SUDO="sudo"
else
  SUDO=""
fi

# --- (1) Base prerequisites ---
echo "Updating package list & installing base tools..."
$SUDO apt-get update -y
# software-properties-common might be needed if using add-apt-repository for Python versions
$SUDO apt-get install -y software-properties-common 
$SUDO apt-get update -y # In case PPA was added (though currently commented out)

$SUDO apt-get install -y vim ca-certificates curl gnupg lsb-release python3.11 python3.11-venv python3.11-dev git nano python3-pip jq rsync tmux build-essential

# --- (2) NVIDIA driver check & install if missing ---
echo "Checking for NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "NVIDIA driver (nvidia-smi) not found. Attempting to install NVIDIA driver 550..."
  # This assumes nvidia-driver-550 is appropriate and available.
  $SUDO apt-get install -y nvidia-driver-550
else
  echo "NVIDIA driver (nvidia-smi) already present."
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk '{print "Driver version: " $1}'
fi

# --- (3) CUDA Toolkit 12.4 Installation ---
echo "Checking for CUDA Toolkit 12.4..."
CUDA_VERSION_TARGET="12.4" # Ensure this matches the deb file name
CUDA_INSTALL_PATH="/usr/local/cuda-${CUDA_VERSION_TARGET}"

if [ -d "$CUDA_INSTALL_PATH" ] && [ -x "$CUDA_INSTALL_PATH/bin/nvcc" ]; then
  echo "CUDA ${CUDA_VERSION_TARGET} installation potentially found at $CUDA_INSTALL_PATH. Verifying version..."
  if "$CUDA_INSTALL_PATH/bin/nvcc" --version | grep -q "release ${CUDA_VERSION_TARGET}"; then
    echo "CUDA version matches target. Skipping installation."
  else
    echo "Existing CUDA installation at $CUDA_INSTALL_PATH does not match target version ${CUDA_VERSION_TARGET}. Proceeding with installation attempt."
    # Consider removing the old version if conflicts are likely: $SUDO rm -rf "$CUDA_INSTALL_PATH"
    install_cuda=true
  fi
else
  install_cuda=true
fi

if [ "${install_cuda:-false}" = true ]; then
  echo "Installing CUDA Toolkit ${CUDA_VERSION_TARGET}..."
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
  $SUDO mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget -q https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
  $SUDO dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
  $SUDO cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
  $SUDO apt-get update -y
  $SUDO apt-get -y install cuda-toolkit-12-4
  
  echo "Updating dynamic linker cache..."
  $SUDO ldconfig
  rm cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb # Clean up installer
fi

# --- (4) Set up CUDA environment variables ---
RCFILE="$HOME/.bashrc"
CUDA_ENV_MARKER="CUDA_PATH_SETUP_V2_FOR_${CUDA_VERSION_TARGET}"
if grep -qF "$CUDA_ENV_MARKER" "$RCFILE"; then
  echo "CUDA ${CUDA_VERSION_TARGET} environment variables already in $RCFILE."
else
  echo "Adding CUDA ${CUDA_VERSION_TARGET} environment variables to $RCFILE..."
  cat <<EOF >> "$RCFILE"

# CUDA ${CUDA_VERSION_TARGET} setup (${CUDA_ENV_MARKER})
export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH}"
export PATH="\${CUDA_INSTALL_PATH}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CUDA_INSTALL_PATH}/lib64:\${LD_LIBRARY_PATH}"
export LIBRARY_PATH="\${CUDA_INSTALL_PATH}/lib64:\${LIBRARY_PATH}"
EOF
fi
# Source for current script session
export CUDA_INSTALL_PATH="$CUDA_INSTALL_PATH"
export PATH="$CUDA_INSTALL_PATH/bin${PATH:+:$PATH}"
export LD_LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_INSTALL_PATH/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"

echo "CUDA environment variables set for current session and added to $RCFILE."
echo "Verifying NVIDIA and CUDA tools..."
nvidia-smi || echo "Warning: nvidia-smi command failed. Check NVIDIA driver installation."
nvcc --version || echo "Warning: nvcc command not found or failed. Check CUDA Toolkit installation and PATH."

# --- (5) Clone or Update tplr-ai-local repository ---
REPO_URL="https://github.com/amiiir-sarfi/tplr-ai-local"
CLONE_DIR="$HOME/tplr-ai-local"
echo "Setting up repository $REPO_URL in $CLONE_DIR..."
if [ -d "$CLONE_DIR/.git" ]; then
  echo "Directory $CLONE_DIR exists and is a git repository. Pulling latest changes..."
  cd "$CLONE_DIR"
  git pull
else
  if [ -d "$CLONE_DIR" ]; then
    echo "Warning: Directory $CLONE_DIR exists but is not a git repository. Removing and re-cloning."
    rm -rf "$CLONE_DIR"
  fi
  echo "Cloning $REPO_URL into $CLONE_DIR..."
  git clone "$REPO_URL" "$CLONE_DIR"
  cd "$CLONE_DIR"
fi

# --- (6) Install uv and Sync Python Environment ---
echo "Ensuring pip and uv are installed/updated for python3.11..."
python3.11 -m pip install --upgrade pip # Ensure pip is up-to-date for uv
python3.11 -m pip install uv

echo "Running 'uv sync' to install Python dependencies from pyproject.toml in $CLONE_DIR..."
# Ensure we are in the correct directory (cd "$CLONE_DIR" above handles this)
python3.11 -m uv sync --all-extras

# --- (7) Setup Python venv PATH for future interactive sessions ---
VENV_PATH="$CLONE_DIR/.venv"
VENV_MARKER="TPLR_AI_LOCAL_VENV_PATH_V1"
if grep -qF "$VENV_MARKER" "$RCFILE"; then
  echo "tplr-ai-local venv PATH already configured in $RCFILE."
else
  echo "Adding tplr-ai-local venv PATH to $RCFILE for interactive sessions..."
  cat <<EOF >> "$RCFILE"

# Add tplr-ai-local Python environment to PATH (${VENV_MARKER})
export PATH="${VENV_PATH}/bin:\${PATH}"
EOF
fi

python3.11 -m pip install huggingface_hub
if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN found, attempting to login to Hugging Face CLI..."
  huggingface-cli login --token "$HF_TOKEN" || echo "huggingface-cli login failed, but continuing..."
fi

echo ""
echo "--- Celium Node Environment Setup Complete ---"