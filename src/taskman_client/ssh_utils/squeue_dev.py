import os
import paramiko
import logging

from taskman_client.ssh_utils.slurm_helper import submit_slurm_job, parse_ls, parse_squeue

# Configure logging
logger = logging.getLogger("SSH")


class RemoteSSHClient:
    def __init__(self, hostname, username, key_file=None, password=None):
        self.base_path = "/home/youngwookim_umass_edu/code/Chair"
        logger.debug("Initializing SSH client")
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Set default key file path if not provided
        default_key_paths = [
            os.path.expanduser('~/.ssh/id_rsa'),
            os.path.expanduser('~/.ssh/id_dsa'),
            os.path.expanduser('~/.ssh/id_ecdsa'),
            os.path.expanduser('~/.ssh/id_ed25519')
        ]

        if key_file is None:
            for key_path in default_key_paths:
                if os.path.exists(key_path):
                    key_file = key_path
                    break

        # Connect using SSH key if available, otherwise use password
        if key_file and os.path.exists(key_file):
            logger.debug("Connecting to SSH with key file")
            self.client.connect(hostname, username=username, key_filename=key_file)
        elif password:
            logger.debug("Connecting to SSH with password")
            self.client.connect(hostname, username=username, password=password)
        else:
            logger.error("SSH connection requires either key_file or password")
            raise ValueError("Either key_file or password must be provided")

    def close(self):
        logger.debug("Closing SSH connection")
        self.client.close()

    def run_command(self, command):
        logger.info(f"Running command: {command}")
        stdin, stdout, stderr = self.client.exec_command(command)
        return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')

    def ls(self, path='.'):
        logger.info(f"Listing directory: {path}")
        output, error = self.run_command(f'ls -l {path}')
        if error:
            logger.error(f"Error in 'ls' command: {error}")
            return error
        return parse_ls(output)

    def squeue(self, user):
        logger.info(f"Fetching squeue for user: {user}")
        output, error = self.run_command(f'squeue -u {user}')
        if error:
            logger.error(f"Error in 'squeue' command: {error}")
            return error
        return parse_squeue(output)

    def submit_slurm_job(self, command, job_name):
        return submit_slurm_job(self.client, command, job_name, self.base_path)
