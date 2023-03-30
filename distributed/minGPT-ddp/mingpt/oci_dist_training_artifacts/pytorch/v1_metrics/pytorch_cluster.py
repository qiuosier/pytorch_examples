import copy
import os
import shlex
import subprocess
from time import time
import torch
import torch.multiprocessing as mp
from ads.opctl.distributed.common.abstract_cluster_provider import ClusterProvider

UNDEFINED_OCID = "Undefined"
class PyTorchProvider(ClusterProvider):
    def __init__(self, mode=None, ephemeral=False, life_span="0h", work_dir=""):
        mode = "MAIN" if str(os.environ["RANK"]) == "0" else "WORKER"
        super().__init__(mode, ephemeral, life_span, work_dir)

    def configuration(self, conf: dict = {}) -> dict:
        config = {
            "MASTER_ADDR" if self.mode == "MAIN" else "OCI__WORKER_IP": self.ip
        }
        return config

    def start(self):
        self.when_ready(self.export_configuration, [self.main_config_file])
        if not "LOCAL_RANK" in os.environ:
            os.environ["LOCAL_RANK"] = "0"

    def find_self_ip(self, authinfo):
        if os.environ["JOB_OCID"] == UNDEFINED_OCID:  # for docker-compose mode
            import socket
            hostname = socket.gethostname()
            IPAddr = socket.gethostbyname(hostname)
            print(f'local setup ip {IPAddr}')
            return IPAddr
        else:
            return super().find_self_ip(authinfo)

    @staticmethod
    def set_env(local_rank):
        gpu_count = torch.cuda.device_count()
        env = copy.deepcopy(os.environ)
        env["NODE_RANK"] = os.environ["RANK"]
        env["LOCAL_RANK"] = str(local_rank)
        env["RANK"] = str(int(env["NODE_RANK"]) * gpu_count + local_rank)
        return env

    @staticmethod
    def run_cmd(cmd, env=None):
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")
        if env:
            print(f"PID: {os.getpid()}, RANK: {env.get('RANK')}", flush=True)
            print(f"PID: {os.getpid()}, LOCAL_RANK: {env.get('LOCAL_RANK')}", flush=True)
        else:
            print(f"RANK: {os.environ['RANK']}", flush=True)

        training_start_time = time()
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            raise Exception("PyTorch distributed errored out...", ret)
        else:
            print("Training Time: ", time() - training_start_time, "seconds")

    @staticmethod
    def run_cmd_multi_gpu(local_rank, cmd):
        env = PyTorchProvider.set_env(local_rank)
        PyTorchProvider.run_cmd(cmd, env=env)

    def run_code(self):
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = "29400"
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
        if 'OCI__WORKER_COUNT' in os.environ:
            os.environ["WORLD_SIZE"] = str(int(os.environ['OCI__WORKER_COUNT']) + 1)

        gpu_count = torch.cuda.device_count()
        print(f"GPU COUNT: {gpu_count}")

        if gpu_count > 1:
            os.environ["WORLD_SIZE"] = str(int(os.environ["WORLD_SIZE"]) * gpu_count)

        code_dir = os.environ.get("OCI__CODE_DIR", "/code")
        training_script = os.path.join(code_dir, os.environ["OCI__ENTRY_SCRIPT"])

        cmd = self.profile_cmd()
        cmd.extend(["python", training_script])
        args = os.environ.get("OCI__ENTRY_SCRIPT_ARGS")
        if args:
            cmd += shlex.split(args)

        print("Running: ", ' '.join(cmd), flush=True)

        if gpu_count > 1:
            mp.spawn(self.run_cmd_multi_gpu, args=(cmd,), nprocs=gpu_count)
        else:
            self.run_cmd(cmd=cmd)
        self.code_execution_complete = True


