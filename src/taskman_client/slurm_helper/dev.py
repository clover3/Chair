

import datetime

from simple_slurm import Slurm
#
# slurm = Slurm(
#     array=range(3, 12),
#     cpus_per_task=2,
#     dependency=dict(after=65541, afterok=34987),
#     gres=['gpu:kepler:2', 'gpu:tesla:2', 'mps:400'],
#     ignore_pbs=True,
#     job_name='name',
#     output=f'{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
#     time=datetime.timedelta(days=1, hours=2, minutes=3, seconds=4),
# )

slurm = Slurm(
    cpus_per_task=2,
    job_name='name',
    output=f'output/log/%j.txt',
    partition='gypsum-1080ti',
    gres=['gpu:1'],
    time=datetime.timedelta(hours=23, minutes=0, seconds=0),
)

slurm.add_cmd("source /work/pi_allan_umass_edu/youngwookim/work/miniconda3/etc/profile.d/conda.sh")
slurm.add_cmd("conda activate tf29")
slurm.add_cmd("source ~/load_cuda.sh")
slurm.sbatch('python demo.py')
slurm.reset_cmd()