import os


def running_on_server():
    return "SLURM_JOB_ID" in os.environ
