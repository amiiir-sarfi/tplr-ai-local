# RentCompute CLI

A command line utility for renting and managing GPU compute instances from cloud providers.

## Installation

```bash
python3.11 -m uv sync --all-extras
```

## Usage Guide

### Setting up API credentials

```bash
rentcompute login
```

This will prompt you to enter your API key, which will be securely stored in `~/.rentcompute/credentials.yaml`.

### Finding Available Instances

```bash
# Search with specific requirements
rentcompute search --gpu-min=2 --gpu-type=h100 --price-max=5

# Filter by name pattern
rentcompute search --name=gpu

# View all available instances
rentcompute search
```

This displays available instances matching your criteria without starting them.

### Starting a Compute Instance

```bash
# Example: Start an H100 instance, provision it, and run a W&B agent
rentcompute start \
--gpu-min=5 --gpu-max=8 \
--gpu-type=h100 \
--price-min=0. --price-max=16 \
--provision \
--ssh-key $HOME/.ssh/id_rsa.pub \
--wandb-agents "sweep_id_1/agent_id_1" "sweep_id_2/agent_id_2" \
--local-env-path $HOME/tplr-ai-local/.env

# Start with specific requirements and custom name
rentcompute start --name="my-gpu-server" --gpu-min=2 --gpu-max=8 --gpu-type=h100 --price-max=5

# Start with specific GPU requirements
rentcompute start --gpu-min=4 --gpu-type=h100

# Start with price constraints
rentcompute start --price-max=3.50

# Start any available instance (lowest cost option)
rentcompute start

# Start and automatically provision using .rentcompute.yml
rentcompute start --gpu-type=h100 --provision
```

After starting, the tool will display SSH connection details for accessing your instance. Checkout `.env.example` and create a `.env` file accordingly.

### Managing Active Instances

List all your active instances:

```bash
rentcompute list
```

This shows all running instances with their details:
- Instance name and ID
- SSH connection details (host, user, port)
- Status
- GPU specifications
- Hourly price
- Ready-to-use SSH command

### Provisioning Instances

Provisioning allows you to automatically configure new or existing instances.

#### Provisioning During Start
```bash
# Start and provision a new instance, potentially running a job
rentcompute start \
--gpu-type=h100 \
--provision \
--wandb-agent "your_wandb_sweep_id/your_agent_id" \
--local-env-path $HOME/tplr-ai-local/.env
```

```bash
# Start and provision a new instance
rentcompute start --gpu-type=h100 --provision
```

#### Provisioning Existing Instances

```bash
# Provision an existing instance (with confirmation)
rentcompute provision --id <instance-id>

# Provision an existing instance and specify a W&B agent to run
rentcompute provision --id <instance-id> \
--wandb-agents "sweep_id_1/agent_id_1" "sweep_id_2/agent_id_2" \
--local-env-path $HOME/tplr-ai-local/.env

# Provision without confirmation
rentcompute provision --id <instance-id> -y
```
When re-provisioning with --wandb-agent, the specified agent will run, and the instance may shut down afterward, depending on your provisioning script's behavior.

#### Provisioning Configuration

Provisioning uses the .rentcompute.yml file in your current directory. Example for script-based provisioning (like run_job_on_celium.sh):

```yaml
# Instance provisioning configuration
provisioning:
  # Type: script, ansible, or docker
  type: script
  # Path to the script that sets up the environment and runs the job
  script: ./run_job_on_celium.sh
```

Supported provisioning methods:
- **ansible**: Runs Ansible playbooks on the instance
- **script**: Executes shell scripts with instance details as environment variables
- **docker**: Copies and runs docker-compose files on the instance

### Syncing Files with Instances

Sync your local files with running instances:

```bash
# Sync with all instances (with confirmation)
rentcompute rsync

# Sync with a specific instance
rentcompute rsync --id <instance-id>

# Sync without confirmation
rentcompute rsync -y

# Sync and reload instances after sync
rentcompute rsync --reload

# Use a custom config file
rentcompute rsync --config custom-config.yml
```

This uses rsync with `-avzP --delete` options and automatically excludes common development directories like `node_modules`, `target`, `venv`, etc.

Sync configuration in `.rentcompute.yml`:

```yaml
# Directories to sync
sync:
  - source: ./data
    destination: ~/data
  - source: ./src
    destination: ~/project/src
  - source: ./scripts
    destination: ~/scripts
```

### Reloading Instances

Reload running instances after making changes:

```bash
# Reload all instances (with confirmation)
rentcompute reload --all

# Reload a specific instance
rentcompute reload --id <instance-id>

# Reload without confirmation
rentcompute reload --all -y

# Use a custom config file
rentcompute reload --all --config custom-config.yml
```

You can also reload instances immediately after syncing files by adding the `--reload` flag to the rsync command:

```bash
rentcompute rsync --reload
```

Reload configuration in `.rentcompute.yml`:

```yaml
# Configuration for reloading instances
reload:
  type: ansible
  playbook: ./reload.yml
  root_dir: ../localnet
  hosts_group: localnet
  vars_file: group_vars/all/vault.yml
  extra_vars:
    remote_mode: true
    gpu_driver: nvidia-latest
```

### Running a Job on an Existing Pod
To run a wandb sweep on an existing pod, use the following command:
```bash
./run_sweep_on_existing_pod.sh -i <POD_ID> -a "sweep_id_1/agent_id_1 sweep_id_2/agent_id_2" [-e <PATH_TO_LOCAL_.ENV_FILE>]
```
To get a list of available pods:
```bash
rentcompute list
# OR
./run_sweep_on_existing_pod.sh
```
This script will:
1. Fetch connection details for the specified pod ID using rentcompute list.
2. Set up necessary environment variables for run_job_on_celium.sh.
3. Execute run_job_on_celium.sh on the remote pod. This includes:
  - Running celium_env_setup.sh for environment preparation.
  - Potentially running scripts/pretokenize_data.py if a local dataset tarball isn't found and the pretokenized data doesn't already exist on the server.
  - Launching the specified W&B agent in the background.

### Stopping Instances

Stop a specific instance:

```bash
# Stop with confirmation
rentcompute stop --id <instance-id>

# Skip confirmation
rentcompute stop --id <instance-id> -y
```

Stop all running instances:

```bash
# Stop all with confirmation
rentcompute stop --all

# Stop all without confirmation
rentcompute stop --all -y
```

The stop command:
1. Verifies instance existence and shows details
2. Asks for confirmation (unless `-y` is used)
3. Sends stop requests to the provider
4. Shows results summary

## Configuration Files

RentCompute uses the following configuration files:

1. **API Credentials**: `~/.rentcompute/credentials.yaml`
   - Contains provider API keys

2. **Instance Configuration**: `.rentcompute.yml` (in working directory)
   - Provisioning configuration
   - Sync directory mappings
   - Environment variables

## Command Reference

| Command | Description |
|---------|-------------|
| `login` | Set API credentials |
| `search` | Find available instances |
| `start` | Start a new instance |
| `list` | List active instances |
| `provision` | Provision an existing instance |
| `rsync` | Sync files with instances |
| `reload` | Reload instances after changes |
| `stop` | Stop instance(s) |

## Development

To set up the development environment:

```bash
pip install -e .
```

Running the tool during development:

```bash
# Using uv run (recommended)
uv run rentcompute [command]

# Enable debugging
uv run rentcompute --debug [command]
```

## Provider Support

Currently supported providers:
- **Celium**: GPU cloud provider
- **Mock**: Local testing provider

---

The `rentcompute` CLI tool is heavily based on and adapted from the `feat/localnet-dev-wallets` branch of the **[tplr-ai/templar](https://github.com/tplr-ai/templar/tree/feat/localnet-dev-wallets)** repository.