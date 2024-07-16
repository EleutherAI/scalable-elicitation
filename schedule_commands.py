import argparse
import subprocess
import time
import json
from pathlib import Path
from multiprocessing import Queue, Process


parser = argparse.ArgumentParser()
parser.add_argument("cmds_file", type=str)
parser.add_argument("--delay-hours", type=float, default=0)
parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7")
args = parser.parse_args()
cmds_file = args.cmds_file
gpu_ids = list(map(int, args.gpu_ids.split(",")))
print(gpu_ids)
# List of commands to execute
commands = open(cmds_file).read().strip().split("\n")
time.sleep(args.delay_hours * 60 * 60)

# File to store GPU usage state
ROOT = ".scheduler"
GPU_FILE_NAME_FORMAT = "gpu{gpu_id}_state_{process}.json"
process = min([i for i in range(100_000) if not any(Path(ROOT, GPU_FILE_NAME_FORMAT.format(gpu_id=gpu_id, process=i)).exists() for gpu_id in range(8))])
GPU_STATE_FILES = {gpu_id: Path(ROOT, GPU_FILE_NAME_FORMAT.format(gpu_id=gpu_id, process=process)) for gpu_id in gpu_ids}

for gpu_id, gpu_state_file in GPU_STATE_FILES.items():
    gpu_state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(gpu_state_file, 'w') as f:
        f.write(json.dumps({str(gpu_id): False}))

def update_gpu_state(gpu_index, in_use=True):
    
    with open(GPU_STATE_FILES[gpu_index], 'r') as f:
        gpu_state = json.load(f)
    
    gpu_state[str(gpu_index)] = in_use
    
    with open(GPU_STATE_FILES[gpu_index], 'w') as f:
        f.write(json.dumps(gpu_state))


def is_gpu_in_use(gpu_index):
    in_use = False
    for feature_file in Path(ROOT).glob(GPU_FILE_NAME_FORMAT.format(gpu_id=gpu_index, process="*")):
        with open(feature_file, 'r') as f:
            gpu_state = json.load(f)
        in_use = in_use or gpu_state.get(str(gpu_index), False)
    return in_use


def run_next_commands(gpu_index, commands: Queue, wait_sec=1):
    while True:
        if commands.empty():
            print(f"No more commands to run on GPU {gpu_index}")
            return
        
        if not is_gpu_in_use(gpu_index):
            # Get the next command from the queue
            command = commands.get()
            command = f"CUDA_VISIBLE_DEVICES={gpu_index} {command}"

            # Run the command as a subprocess
            try:
                update_gpu_state(gpu_index, True)
                print(f"Running command: {command} on GPU {gpu_index}")
                subprocess.run(command, shell=True, check=False)
            finally:
                update_gpu_state(gpu_index, False)
        
        time.sleep(wait_sec)
        

try:
    # Create a queue to hold the commands
    command_queue = Queue()
    for cmd in commands:
        command_queue.put(cmd)

    # Create and start a process for each GPU
    processes = []
    for gpu_id in gpu_ids:
        process = Process(target=run_next_commands, args=(gpu_id, command_queue))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

    print("All commands have been executed.")
finally:
    for gpu_state_file in GPU_STATE_FILES.values():
        if gpu_state_file.exists():
            gpu_state_file.unlink()