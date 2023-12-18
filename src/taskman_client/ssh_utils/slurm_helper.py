import re
import subprocess

from trainer_v2.chair_logging import c_log


def submit_slurm_job(client, command, job_name, base_path, option_str=""):
    c_log.info("Submitting a new Slurm job")

    # Form paths using base_path
    output_path = f"{base_path}/output/log/%j.txt"
    script_path = f"{base_path}/script_temp/{job_name}.sh"

    # Create a temporary script with Slurm directives
    script_content = f"""#!/bin/bash
#
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}  # output file
#SBATCH -e {output_path}        # File to which STDERR will be written
#
#SBATCH --ntasks=1
#SBATCH --time=24:00:00         # Runtime in D-HH:MM

{command}
"""

    # Function to execute a command either locally or remotely
    def execute_command(exec_command):
        if client:
            # Remote execution
            stdin, stdout, stderr = client.client.exec_command(exec_command)
            return stdout.read().decode('utf-8'), stderr.read().decode('utf-8')
        else:
            # Local execution
            process = subprocess.run(exec_command, shell=True, capture_output=True, text=True)
            return process.stdout, process.stderr

    # Create and write the script
    output, error = execute_command(f"echo '{script_content}' > {script_path}")

    # Check for errors in script creation
    if error:
        c_log.error(f"Error while creating temporary script: {error}")
        return None, error

    # Submit the job using sbatch
    output, error = execute_command(f"sbatch {option_str} {script_path}")

    # Parse the job ID from the output
    match = re.search(r"Submitted batch job (\d+)", output)
    if match:
        job_id = match.group(1)
        c_log.info(f"Slurm job submitted successfully with Job ID: {job_id}")
        return job_id, None
    else:
        error_msg = "Unable to parse job ID from sbatch output."
        c_log.error(error_msg)
        return None, error_msg


def parse_ls(output):
    lines = output.strip().split('\n')
    parsed_data = []
    for line in lines:
        parts = line.split()
        if len(parts) < 8:
            continue  # Skip malformed lines
        file_info = {
            'permissions': parts[0],
            'links': int(parts[1]),
            'user': parts[2],
            'group': parts[3],
            'size': int(parts[4]),
            'time_str': ' '.join(parts[5:-1]),
            'filename': parts[-1]
        }
        parsed_data.append(file_info)
    return parsed_data


def parse_table(output):
    lines = output.strip().split('\n')
    headers = lines[0].split()
    data = [line.split() for line in lines[1:]]
    return headers, data


def parse_squeue(output):
    headers, data = parse_table(output)
    parsed_data = []
    for row in data:
        row_dict = {headers[i]: row[i] for i in range(len(headers))}
        parsed_data.append(row_dict)
    return parsed_data
