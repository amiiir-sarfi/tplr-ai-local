"""
Provision command implementation.
"""

import logging
from typing import Optional, Dict, Any, List

from rentcompute.config import Config
from rentcompute.provisioning import provision_instance

logger = logging.getLogger(__name__)


def run(
    config: Config, 
    instance_id: str, 
    skip_confirmation: bool = False,
    wandb_agents: Optional[List[str]] = None, # Changed from wandb_agent, now a list
    local_env_path: Optional[str] = None    
) -> None:
    """Run the provision command to provision an existing instance.

    Args:
        config: Configuration manager
        instance_id: ID of the instance to provision
        skip_confirmation: Whether to skip the confirmation prompt
        wandb_agents: Optional list of W&B agent IDs to run (passed to provisioning script).
        local_env_path: Optional path to local .env file (passed to provisioning script).
    """
    provider = config.get_provider()

    print(f"Looking up instance with ID: {instance_id}...")

    pods = provider.list_pods()
    target_pod = None

    for pod in pods:
        if pod.id == instance_id:
            target_pod = pod
            break

    if not target_pod:
        print(f"Error: No active instance found with ID {instance_id}")
        print("Use 'rentcompute list' to see active instances")
        return

    print(f"Found instance '{target_pod.name}' (ID: {target_pod.id})")
    print(f"Host: {target_pod.host}, Port: {target_pod.port}")
    print(f"GPU: {target_pod.gpu_count}x {target_pod.gpu_type}")
    print(f"Hourly rate: ${target_pod.hourly_rate:.2f}/hr")

    if not skip_confirmation:
        confirm_message = "Provision this instance?"
        if wandb_agents and len(wandb_agents) > 0:
            confirm_message += f" This will also attempt to run W&B agent(s) '{', '.join(wandb_agents)}' sequentially."
        confirm_message += " (y/n): "
        confirm = input(confirm_message)
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return
    else:
        print("Skipping confirmation due to -y/--yes flag.")
        if wandb_agents and len(wandb_agents) > 0:
            print(f"Will attempt to run W&B agent(s) '{', '.join(wandb_agents)}' sequentially as part of provisioning.")

    instance_config_for_script: Optional[Dict[str, Any]] = None
    # Check if wandb_agents list is not None and not empty, or if local_env_path is provided
    if (wandb_agents and len(wandb_agents) > 0) or local_env_path:
        instance_config_for_script = {
            "wandb_agents": wandb_agents if wandb_agents else [], # Ensure it's a list
            "local_env_path": local_env_path
        }
        logger.debug(f"Passing instance_config_for_script to provisioning: {instance_config_for_script}")


    print(
        "\nProvisioning requested. Looking for .rentcompute.yml in current directory..."
    )
    if provision_instance(target_pod, instance_config_for_script=instance_config_for_script):
        print("Provisioning process completed.") 

        private_key_path = (
            target_pod.key_path.replace(".pub", "")
            if target_pod.key_path.endswith(".pub")
            else target_pod.key_path
        )
        print("\nSSH Connection Details:")
        print(f"Host: {target_pod.host}")
        print(f"User: {target_pod.user}")
        print(f"Port: {target_pod.port}")
        print(f"SSH Key: {private_key_path}")
        print("\nConnection command:")
        print(
            f"ssh {target_pod.user}@{target_pod.host} -p {target_pod.port} -i {private_key_path}"
        )
    else:
        print(
            "Provisioning script wrapper indicated failure. Instance is still running but may require manual setup or re-provisioning."
        )