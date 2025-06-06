import os
import shutil
import torch.distributed as torch_dist
import subprocess

def get_checkpoint_list(checkpoint_dir):
    dir_steps_list = []
    dir_list = os.listdir(checkpoint_dir)
    if len(dir_list) > 0:
        for _dir in dir_list:
            completed_steps = int(_dir.split('_')[1])
            loader_steps = int(_dir.split('_')[2])
            dir_steps_list.append([_dir, completed_steps, loader_steps])
        dir_steps_list = sorted(dir_steps_list, key=lambda x: x[1])
    else:
        dir_steps_list = [['', 0, 0]]
    return dir_steps_list


def clean_checkpoint_folder(checkpoint_dir, max_keep=1):
    # assert max_keep > 0, "max_keep should be greater than 0"
    dir_steps_list = get_checkpoint_list(checkpoint_dir)
    if len(dir_steps_list) >= max_keep and dir_steps_list[0][0] != '':
        for _dir, _, _ in dir_steps_list[:len(dir_steps_list)-max_keep]:
            shutil.rmtree(os.path.join(checkpoint_dir, _dir))



import debugpy
from termcolor import colored

# def setup_debugpy(accelerator, endpoint="localhost", port=5678, rank=0, force=False):
#     if "DEBUGPY" not in os.environ:
#         print(colored(f"DEBUGPY not in os.environ", "red"))
#         return
#     rank = int(os.getenv("DEBUGPY_RANK", rank))
#     port = int(os.getenv("DEBUGPY_PORT", port))
#     endpoint = os.getenv("DEBUGPY_ENDPOINT", endpoint)
#     if accelerator.process_index != rank:
#         accelerator.wait_for_everyone()
#         return
#     # print(colored(f"rank: {get_rank()}, is_main_process: {is_main_process()}", "red"))
#     if force:
#         # run_cmd("ps aux | grep debugpy | awk '{print $2}' | xargs kill -9", fault_tolerance=True)
#         print(debugpy(f"Force killed debugpy", "red"))
#     try:
#         debugpy.listen((endpoint, port))
#         print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
#         debugpy.wait_for_client()
#     except:
#         print(colored(f"Failed to setup debugpy, {endpoint}:{port} occupied", "red"))

#     accelerator.wait_for_everyone()

def run_cmd(cmd, verbose=False, async_cmd=False, conda_env=None, fault_tolerance=False):
    if conda_env is not None:
        cmd = f"conda run -n {conda_env} {cmd}"

    if verbose:
        assert not async_cmd, "async_cmd is not supported when verbose=True"
        popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in popen.stdout:
            print(line.rstrip().decode("utf-8"))
        popen.wait()
        if popen.returncode != 0 and not fault_tolerance:
            raise RuntimeError(f"Failed to run command: {cmd}\nERROR {popen.stderr}\nSTDOUT{popen.stdout}")
        return popen.returncode
    else:
        if not async_cmd:
            # decode bug fix: https://stackoverflow.com/questions/73545218/utf-8-encoding-exception-with-subprocess-run
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding="cp437")
            if ret.returncode != 0 and not fault_tolerance:
                raise RuntimeError(f"Failed to run command: {cmd}\nERROR {ret.stderr}\nSTDOUT{ret.stdout}")
            return ret
        else:
            popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return popen

def setup_debugpy(endpoint="localhost", port=5678, rank=0, force=False):
    if "DEBUGPY" not in os.environ:
        return
    rank = int(os.getenv("DEBUGPY_RANK", rank))
    port = int(os.getenv("DEBUGPY_PORT", port))
    endpoint = os.getenv("DEBUGPY_ENDPOINT", endpoint)
    if get_rank() != rank:
        synchronize()
        return

    # print(colored(f"rank: {get_rank()}, is_main_process: {is_main_process()}", "red"))
    if force:
        run_cmd("ps aux | grep /debugpy/adapter | awk '{print $2}' | xargs kill -9", fault_tolerance=True)
        print(colored("Force killed debugpy", "red"))
        
    try:
        debugpy.listen((endpoint, port))
        print(colored(f"Waiting for debugger attach on {endpoint}:{port}", "red"))
        debugpy.wait_for_client()
    except:
        print(colored(f"Failed to setup debugpy, {endpoint}:{port} occupied", "red"))

    synchronize()

def get_rank() -> int:
    if not torch_dist.is_available():
        return 0
    if not torch_dist.is_initialized():
        return 0
    return torch_dist.get_rank()

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not torch_dist.is_available():
        return
    if not torch_dist.is_initialized():
        return
    world_size = torch_dist.get_world_size()
    if world_size == 1:
        return

    # if torch_dist.get_backend() == torch_dist.Backend.NCCL:
    #     # This argument is needed to avoid warnings.
    #     # It's valid only for NCCL backend.
    #     torch_dist.barrier(device_ids=[torch.cuda.current_device()])
    # else:
    #     torch_dist.barrier()
    torch_dist.barrier()