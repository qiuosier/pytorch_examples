"""Code to launch distributed training on single machine with multiple-GPUs
"""
import copy
import os
import subprocess
import sys
from time import time

import torch
import torch.multiprocessing as mp


def run_training(rank):
    env = copy.deepcopy(os.environ)
    env["RANK"] = str(rank)
    env["LOCAL_RANK"] = str(rank)
    print(f"GPU-{env.get('RANK')}: PID={os.getpid()}", flush=True)

    cmd = ["python", os.path.join(os.path.dirname(__file__), "main.py")]
    cmd += sys.argv[1:]

    training_start_time = time()
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise Exception(f"GPU-{env.get('RANK')}: exit code {ret}", ret)
    else:
        print("Training Time: ", time() - training_start_time, "seconds")


def main():
    # hostname = socket.gethostname()
    # ip_address = socket.gethostbyname(hostname)
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "29400"
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise EnvironmentError("GPU not available.")
    else:
        print(f"GPU COUNT: {gpu_count}")
    os.environ["WORLD_SIZE"] = str(gpu_count)
    mp.spawn(run_training, nprocs=gpu_count)


if __name__ == "__main__":
    main()
