"""
Provisioning module for rentcompute.

This module handles instance provisioning based on .rentcompute.yml configuration.
"""

import os
import subprocess
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from rentcompute.providers.base import Pod

# Configure logger
logger = logging.getLogger(__name__)


class ProvisioningConfig:
    """Configuration for instance provisioning."""

    def __init__(self, config_path: str = ".rentcompute.yml") -> None:
        """Initialize provisioning configuration.

        Args:
            config_path: Path to the provisioning config file
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.valid = False

    def load(self) -> bool:
        """Load provisioning configuration from file.

        Returns:
            True if loading was successful, False otherwise
        """
        if not os.path.exists(self.config_path):
            logger.error(f"Provisioning config file not found: {self.config_path}")
            return False

        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
                if not self.config:
                    logger.error("Empty provisioning config file")
                    return False

                self.valid = self._validate_config()
                return self.valid
        except Exception as e:
            logger.error(f"Error loading provisioning config: {e}")
            return False

    def _validate_config(self) -> bool:
        """Validate the loaded configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check for required sections
        if "provisioning" not in self.config:
            logger.error("Missing 'provisioning' section in config")
            return False

        provisioning = self.config["provisioning"]

        # Check for required provisioning type
        if "type" not in provisioning:
            logger.error("Missing 'type' in provisioning section")
            return False

        # Validate based on provisioning type
        prov_type = provisioning["type"].lower()

        if prov_type == "script":
            # Validate script configuration
            if "script" not in provisioning:
                logger.error("Missing 'script' in provisioning section")
                return False
        elif prov_type == "ansible":
            # Validate ansible configuration
            if "playbook" not in provisioning:
                logger.error("Missing 'playbook' in provisioning section")
                return False

            # Check if root_dir exists if specified
            if "root_dir" in provisioning:
                root_dir = provisioning["root_dir"]
                if root_dir:
                    try:
                        # First expand any ~ in the path
                        expanded_path = os.path.expanduser(root_dir)

                        # Create a Path object and resolve to absolute path
                        root_path = Path(expanded_path).resolve()

                        if not root_path.is_dir():
                            logger.warning(
                                f"Ansible root_dir '{root_path}' is not a valid directory"
                            )
                            # This is just a warning, not an error, as we'll fall back to current directory
                    except Exception as e:
                        logger.warning(f"Error resolving Ansible root_dir path: {e}")
                        # This is just a warning, not an error, as we'll fall back to current directory
        elif prov_type == "docker":
            # Validate docker configuration
            if "compose_file" not in provisioning:
                logger.error("Missing 'compose_file' in provisioning section")
                return False
        else:
            logger.error(f"Unsupported provisioning type: {prov_type}")
            return False

        return True


def provision_instance(pod: Pod, instance_config_for_script: Optional[Dict[str, Any]] = None) -> bool:
    """Provision an instance based on local configuration.

    Args:
        pod: Instance pod to provision
        instance_config_for_script: Configuration for the instance script (optional),
                                    may include 'wandb_agents' (list) and 'local_env_path'.

    Returns:
        True if provisioning was successful, False otherwise
    """
    logger.info(f"Provisioning instance {pod.id} ({pod.name})...")

    # Load provisioning configuration
    config = ProvisioningConfig()
    if not config.load():
        print("Failed to load provisioning configuration from .rentcompute.yml")
        return False

    # Get provisioning type and configuration
    prov_config = config.config["provisioning"]
    prov_type = prov_config["type"].lower()

    try:
        if prov_type == "script":
            return _provision_with_script(pod, prov_config, instance_config_for_script)
        elif prov_type == "ansible":
            return _provision_with_ansible(pod, prov_config)
        elif prov_type == "docker":
            return _provision_with_docker(pod, prov_config)
        else:
            print(f"Unsupported provisioning type: {prov_type}")
            return False
    except Exception as e:
        logger.error(f"Provisioning error: {e}")
        print(f"Error during provisioning: {e}")
        return False


def _provision_with_script(pod: Pod, config: Dict[str, Any], instance_config_for_script: Optional[Dict[str, Any]] = None) -> bool:
    """Provision instance using a script.

    Args:
        pod: Instance pod to provision
        config: Provisioning configuration
        instance_config_for_script: Optional dict, may contain 'wandb_agents' (list) and 'local_env_path'.

    Returns:
        True if provisioning was successful, False otherwise
    """
    script_path = config.get("script")
    if not script_path:
        print("No script specified in provisioning configuration")
        return False

    # Check if script exists
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False

    # Make script executable
    try:
        os.chmod(script_path, 0o755)
    except Exception as e:
        print(f"Failed to make script executable: {e}")
        return False

    # Execute script with pod details as environment variables
    env = os.environ.copy()
    env.update(
        {
            "RENTCOMPUTE_POD_ID": pod.id,
            "RENTCOMPUTE_POD_NAME": pod.name,
            "RENTCOMPUTE_POD_HOST": pod.host,
            "RENTCOMPUTE_POD_USER": pod.user,
            "RENTCOMPUTE_POD_PORT": str(pod.port),
            "RENTCOMPUTE_POD_KEY": pod.key_path,
        }
    )
    if instance_config_for_script:
        wandb_agents_list: Optional[List[str]] = instance_config_for_script.get("wandb_agents")
        if wandb_agents_list and isinstance(wandb_agents_list, list) and len(wandb_agents_list) > 0:
            env["RENTCOMPUTE_WANDB_AGENT_LIST"] = ",".join(wandb_agents_list)
            logger.info(f"Passing RENTCOMPUTE_WANDB_AGENT_LIST: {env['RENTCOMPUTE_WANDB_AGENT_LIST']}")
        
        local_env_path_raw = instance_config_for_script.get("local_env_path")
        if local_env_path_raw:
            local_env_path = os.path.expanduser(local_env_path_raw)
            if os.path.exists(local_env_path):
                env["RENTCOMPUTE_LOCAL_ENV_PATH"] = local_env_path
            else:
                logger.warning(f"Local .env path specified but not found: {local_env_path}")

    print(f"Running provisioning script: {script_path}")
    process = subprocess.Popen(
        [script_path], 
        env=env, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        bufsize=1 
    )
    
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            print(line, end='') 
    
    return_code = process.wait()


    if return_code == 0:
        print("Provisioning script wrapper executed successfully.")
        return True
    else:
        print(f"Provisioning script wrapper failed with exit code {return_code}.")
        return False


def _provision_with_ansible(pod: Pod, config: Dict[str, Any]) -> bool:
    """Provision instance using Ansible.

    Args:
        pod: Instance pod to provision
        config: Provisioning configuration

    Returns:
        True if provisioning was successful, False otherwise
    """
    playbook_path = config.get("playbook")
    if not playbook_path:
        print("No playbook specified in provisioning configuration")
        return False

    current_dir = os.getcwd()
    root_dir = config.get("root_dir")
    ansible_dir = current_dir

    if root_dir:
        try:
            expanded_path = os.path.expanduser(root_dir)
            root_path = Path(expanded_path)
            resolved_path = root_path.resolve()
            if resolved_path.is_dir():
                ansible_dir = str(resolved_path)
                print(f"Using Ansible root directory: {ansible_dir}")
                if not os.path.isabs(playbook_path):
                    playbook_path = os.path.join(ansible_dir, playbook_path)
            else:
                print(
                    f"Warning: Specified Ansible root directory '{resolved_path}' not found. Using current directory."
                )
        except Exception as e:
            print(
                f"Error resolving Ansible root directory: {e}. Using current directory."
            )

    if not os.path.exists(playbook_path):
        print(f"Ansible playbook not found: {playbook_path}")
        return False

    inventory_path = Path(ansible_dir) / ".rentcompute_inventory.ini"
    private_key_path = (
        pod.key_path.replace(".pub", "")
        if pod.key_path.endswith(".pub")
        else pod.key_path
    )

    try:
        hosts_group = config.get("hosts_group", "rentcompute")
        with open(inventory_path, "w") as f:
            f.write(f"[{hosts_group}]\n")
            f.write(
                f"{pod.host} ansible_user={pod.user} ansible_port={pod.port} ansible_ssh_private_key_file={private_key_path}\n"
            )

        extra_vars_args = []
        vars_file = config.get("vars_file")
        if vars_file:
            if not os.path.isabs(vars_file):
                vars_file_path = os.path.join(ansible_dir, vars_file)
            else:
                vars_file_path = vars_file
            if os.path.isfile(vars_file_path):
                print(f"Using vars file: {vars_file_path}")
                extra_vars_args.extend(["-e", f"@{vars_file_path}"])
            else:
                print(f"Warning: Vars file not found: {vars_file_path}")

        extra_vars = config.get("extra_vars", {})
        if extra_vars:
            for key, value in extra_vars.items():
                extra_vars_args.extend(["-e", f"{key}={value}"])

        cmd = ["ansible-playbook", "-i", str(inventory_path), playbook_path, "-v"]
        cmd.extend(extra_vars_args)

        original_dir = os.getcwd()
        try:
            if ansible_dir != original_dir:
                os.chdir(ansible_dir)
                print(f"Changed to directory: {ansible_dir}")

            env = os.environ.copy()
            env["ANSIBLE_HOST_KEY_CHECKING"] = "false"
            print(f"Running Ansible provisioning with playbook: {playbook_path}")
            print("\n--- Ansible Provisioning Output ---")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in iter(process.stdout.readline, ""):
                print(line, end="")

            process.stdout.close()
            return_code = process.wait()
            print("--- End of Ansible Output ---\n")

            if return_code == 0:
                print("Ansible provisioning completed successfully")
                return True
            else:
                print(f"Ansible provisioning failed with exit code {return_code}")
                return False
        finally:
            if ansible_dir != original_dir:
                os.chdir(original_dir)
    except Exception as e:
        print(f"Error during Ansible provisioning: {e}")
        return False
    finally:
        if inventory_path.exists():
            inventory_path.unlink()


def _provision_with_docker(pod: Pod, config: Dict[str, Any]) -> bool:
    """Provision instance using Docker.

    Args:
        pod: Instance pod to provision
        config: Provisioning configuration

    Returns:
        True if provisioning was successful, False otherwise
    """
    compose_file = config.get("compose_file")
    if not compose_file:
        print("No docker-compose file specified in provisioning configuration")
        return False

    if not os.path.exists(compose_file):
        print(f"Docker Compose file not found: {compose_file}")
        return False

    private_key_path = (
        pod.key_path.replace(".pub", "")
        if pod.key_path.endswith(".pub")
        else pod.key_path
    )

    scp_cmd = [
        "scp",
        "-i",
        private_key_path,
        "-P",
        str(pod.port),
        compose_file,
        f"{pod.user}@{pod.host}:~/docker-compose.yml",
    ]

    print("Copying Docker Compose file to instance...")
    scp_result = subprocess.run(scp_cmd, capture_output=True, text=True)

    if scp_result.returncode != 0:
        print(f"Failed to copy Docker Compose file: {scp_result.stderr}")
        return False

    ssh_cmd = [
        "ssh",
        "-i",
        private_key_path,
        "-p",
        str(pod.port),
        f"{pod.user}@{pod.host}",
        "docker-compose up -d",
    ]

    print("Running Docker Compose on instance...")
    ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True)

    if ssh_result.returncode == 0:
        print("Docker provisioning completed successfully")
        return True
    else:
        print(f"Docker provisioning failed: {ssh_result.stderr}")
        return False
