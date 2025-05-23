"""
Start command implementation.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from rentcompute.config import Config
from rentcompute.providers.base import Machine
from rentcompute.provisioning import provision_instance

logger = logging.getLogger(__name__)


def run(config: Config, instance_config: Dict[str, Any], name: str = None) -> None:
    """Run the start command.

    Args:
        config: Configuration manager
        instance_config: Configuration including GPU and price requirements,
                         and potentially 'wandb_agents' (list) and 'local_env_path'.
        name: Optional custom name for the instance
    """
    provider = config.get_provider()
    machine_id = instance_config.get("machine_id")
    selected_machine = None

    if machine_id:
        print(f"Starting specific machine with ID: {machine_id}")
        machine_details = provider._get_machine_details(machine_id) # type: ignore # Celium specific method

        if machine_details:
            print(
                f"Found machine: {machine_details.name}, {machine_details.gpu_type}, {machine_details.gpu_count} GPU(s), ${machine_details.hourly_rate:.2f}/hour"
            )
            selected_machine = machine_details
        else:
            print("Machine details not found. Starting with provided ID only.")
            selected_machine = Machine(
                id=machine_id,
                name=f"machine-{machine_id[-8:]}",
                provider_name=provider.name,
                gpu_type="",
                gpu_count=0,
                hourly_rate=0.0,
            )
    else:
        gpu_config = instance_config["gpu"]
        price_config = instance_config["price"]
        min_gpus = gpu_config["min"]
        max_gpus = gpu_config["max"]
        gpu_type = gpu_config["type"]
        min_price = price_config["min"]
        max_price = price_config["max"]

        gpu_filters = [min_gpus, max_gpus, gpu_type]
        price_filters = [min_price, max_price]
        all_filters = gpu_filters + price_filters

        if all(v is None for v in all_filters):
            print(
                f"Starting a new instance from {provider.name} with next available configuration..."
            )
        else:
            filter_desc = []
            gpu_desc = []
            if min_gpus is not None and max_gpus is not None:
                if min_gpus == max_gpus:
                    gpu_desc.append(f"{min_gpus} GPUs")
                else:
                    gpu_desc.append(f"{min_gpus}-{max_gpus} GPUs")
            elif min_gpus is not None:
                gpu_desc.append(f"at least {min_gpus} GPUs")
            elif max_gpus is not None:
                gpu_desc.append(f"up to {max_gpus} GPUs")
            if gpu_type is not None:
                gpu_desc.append(f"type {gpu_type}")
            if gpu_desc:
                filter_desc.append(f"GPU: {' '.join(gpu_desc)}")

            price_desc = []
            if min_price is not None and max_price is not None:
                if min_price == max_price:
                    price_desc.append(f"${min_price}/hr")
                else:
                    price_desc.append(f"${min_price}-${max_price}/hr")
            elif min_price is not None:
                price_desc.append(f"at least ${min_price}/hr")
            elif max_price is not None:
                price_desc.append(f"up to ${max_price}/hr")
            if price_desc:
                filter_desc.append(f"Price: {' '.join(price_desc)}")
            print(
                f"Starting a new instance from {provider.name} with {', '.join(filter_desc)}..."
            )

        available_machines = provider.search_machines(instance_config, name)
        if not available_machines:
            print("No matching machines found. Please try different criteria.")
            return
        selected_machine = _select_machine(available_machines)

    if not selected_machine:
        print("Operation cancelled.")
        return

    ssh_key_path = instance_config.get("ssh_key")
    if ssh_key_path:
        ssh_key_path = os.path.expanduser(ssh_key_path)
        if not os.path.isfile(ssh_key_path):
            print(f"SSH key file not found: {ssh_key_path}")
            return
        logger.debug(f"Using SSH key from: {ssh_key_path}")
    else:
        default_key_path = os.path.expanduser("~/.ssh/id_rsa.pub")
        if os.path.isfile(default_key_path):
            ssh_key_path = default_key_path
            logger.debug(f"Using default SSH key from: {ssh_key_path}")
        else:
            print("No SSH key provided and default ~/.ssh/id_rsa.pub not found.")
            print("Please provide a valid SSH key with --ssh-key.")
            return

    print(f"Starting {selected_machine.name} ({selected_machine.id})...")
    pod = provider.start_machine(selected_machine.id, name, ssh_key_path=ssh_key_path)

    if not pod:
        print("Failed to start the machine.")
        return

    print(f"Instance '{pod.name}' (ID: {pod.id}) started successfully.")
    print(f"Hourly rate: ${pod.hourly_rate:.2f}/hr")
    private_key_path = (
        pod.key_path.replace(".pub", "")
        if pod.key_path.endswith(".pub")
        else pod.key_path
    )
    print(
        f"SSH connection: ssh {pod.user}@{pod.host} -p {pod.port} -i {private_key_path}"
    )

    if instance_config.get("provision"):
        print(
            "\nProvisioning requested. Looking for .rentcompute.yml in current directory..."
        )
        # Pass the full instance_config which now includes 'wandb_agents' and 'local_env_path'
        if provision_instance(pod, instance_config):
            print("Provisioning completed successfully.")
            print("\nSSH Connection Details:")
            print(f"Host: {pod.host}")
            print(f"User: {pod.user}")
            print(f"Port: {pod.port}")
            print(f"SSH Key: {private_key_path}")
            print("\nConnection command:")
            print(f"ssh {pod.user}@{pod.host} -p {pod.port} -i {private_key_path}")
        else:
            print(
                "Provisioning failed. Instance is still running but may require manual setup."
            )


def _select_machine(machines: List[Machine]) -> Optional[Machine]:
    """Select the cheapest machine from the available options.

    Args:
        machines: List of available machines

    Returns:
        Selected machine (cheapest option) or None if no machines
    """
    if not machines:
        return None

    cheapest = min(machines, key=lambda m: m.hourly_rate)
    print(
        f"Selected lowest cost option: {cheapest.name} at ${cheapest.hourly_rate:.2f}/hr"
    )
    return cheapest