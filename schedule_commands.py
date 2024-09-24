import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("cmds_file", type=str)
parser.add_argument("--read-forever", action="store_true")
parser.add_argument("--delay-hours", type=float, default=0)
parser.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7")
args = parser.parse_args()
cmds_file = args.cmds_file
gpu_ids = list(map(int, args.gpu_ids.split(",")))
print(gpu_ids)
datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print(datetime)
# List of commands to execute
commands = open(cmds_file).read().strip().split("\n")

time.sleep(args.delay_hours * 60 * 60)

# File to store GPU usage state
ROOT = Path(__file__).parent / ".scheduler"
LOG_FORMAT = "logs/log_{datetime}_{gpu_id}.txt"
GPU_FILE_NAME_FORMAT = "gpu{gpu_id}_state_{process_id}.json"
PROGRESS_FILE_NAME_FORMAT = "progress_{process_id}.json"

process_id = min(
    [
        i
        for i in range(100_000)
        if not any(
            Path(
                ROOT, GPU_FILE_NAME_FORMAT.format(gpu_id=gpu_id, process_id=i)
            ).exists()
            for gpu_id in range(8)
        )
    ]
)
print(f"Process ID: {process_id}")
GPU_STATE_FILES = {
    gpu_id: Path(
        ROOT, GPU_FILE_NAME_FORMAT.format(gpu_id=gpu_id, process_id=process_id)
    )
    for gpu_id in gpu_ids
}

for gpu_id, gpu_state_file in GPU_STATE_FILES.items():
    gpu_state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(gpu_state_file, "w") as f:
        f.write(json.dumps({str(gpu_id): False}))


def update_gpu_state(gpu_index, in_use=True):
    with open(GPU_STATE_FILES[gpu_index], "r") as f:
        gpu_state = json.load(f)

    gpu_state[str(gpu_index)] = in_use

    with open(GPU_STATE_FILES[gpu_index], "w") as f:
        f.write(json.dumps(gpu_state))


def is_gpu_in_use(gpu_id):
    in_use = False
    for feature_file in Path(ROOT).glob(
        GPU_FILE_NAME_FORMAT.format(gpu_id=gpu_id, process_id="*")
    ):
        with open(feature_file, "r") as f:
            gpu_state = json.load(f)
        in_use = in_use or gpu_state.get(str(gpu_id), False)
    return in_use


def write_progress_file(num_launched: int, of: int):
    with open(
        Path(ROOT, PROGRESS_FILE_NAME_FORMAT.format(process_id=process_id)), "w"
    ) as f:
        f.write(
            json.dumps(
                {"num_launched": num_launched, "of": of, "progress": num_launched / of}
            )
        )


def run_next_commands(gpu_id, commands: Queue, finished_commands: Queue, wait_sec=1):
    while True:
        if commands.empty():
            print(f"No more commands to run on GPU {gpu_id}")
            if args.read_forever:
                time.sleep(10)
                continue
            return

        if not is_gpu_in_use(gpu_id):
            # Pop the next command from the queue
            command = commands.get()
            finished_commands.put(command)
            log_file = Path(ROOT) / LOG_FORMAT.format(datetime=datetime, gpu_id=gpu_id)
            # Create the log file if it doesn't exist
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.touch(exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"Running command: {command} on GPU {gpu_id}\n")
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command} >> {log_file} 2>&1"

            # Run the command as a subprocess
            try:
                update_gpu_state(gpu_id, True)
                write_progress_file(
                    num_launched=finished_commands.qsize(),
                    of=commands.qsize() + finished_commands.qsize(),
                )
                print(f"Running command: {command} on GPU {gpu_id}")
                subprocess.run(command, shell=True, check=False)
            finally:
                update_gpu_state(gpu_id, False)

        time.sleep(wait_sec)


def read_forever(command_queue: Queue, commands: list):
    while True:
        with open(cmds_file, "r") as f:
            # read the new lines
            new_lines = f.read().split("\n")[len(commands) :]

        commands.extend(new_lines)
        for cmd in new_lines:
            command_queue.put(cmd)
        time.sleep(1)


try:
    # Create a queue to hold the commands
    command_queue = Queue()
    for cmd in commands:
        command_queue.put(cmd)

    # Create and start a process for each GPU
    processes = []
    for gpu_id in gpu_ids:
        process = Process(
            target=run_next_commands, args=(gpu_id, command_queue, Queue())
        )
        processes.append(process)
        process.start()

    if args.read_forever:
        process = Process(target=read_forever, args=(command_queue, commands))
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

    # Remove progress file if it exists
    if os.path.exists(
        Path(ROOT, PROGRESS_FILE_NAME_FORMAT.format(process_id=process_id))
    ):
        os.remove(Path(ROOT, PROGRESS_FILE_NAME_FORMAT.format(process_id=process_id)))
