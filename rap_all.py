import os
import subprocess
import multiprocessing
import sys

import torch

GPUS = [i for i in range(torch.cuda.device_count())]

TASKS = []

names = ["court"]
for name in names:
    TASKS.append((name, f"configs/Cambridge/{name}.txt", f"output/Cambridge/{name}", ""))

# TASKS.append(("aachen", "configs/aachen.txt", "output/aachen", ""))
# TASKS.append(("new_church", "configs/new_church.txt", "output/new_church", ""))
#
# names = ["11", "15", "37", "41"]
# for name in names:
#     TASKS.append((name, f"configs/MARS/{name}.txt", f"output/MARS/{name}", ""))

names = ["office", "pumpkin", "kitchen", "stairs"]
# for name in names:
#     TASKS.append((name, f"configs/7Scenes/{name}.txt", f"output/7Scenes/{name}", ""))
for name in names:
    TASKS.append((name, f"configs/7Scenes_sfm/{name}.txt", f"output/7Scenes_sfm/{name}", ""))

LOG_DIR = "logs_out"
os.makedirs(LOG_DIR, exist_ok=True)


def worker_process(gpu_id, task_queue):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[GPU {gpu_id}] No more tasks. Exiting.")
            break

        name, config_path, gs_dir, extra_args = task
        print(f"[GPU {gpu_id}] Starting: {name}")

        cmd = [
            sys.executable, "rap.py",
            "-c", config_path,
            "-m", gs_dir,
        ]
        if extra_args:
            cmd.extend(extra_args.split())

        log_file = os.path.join(LOG_DIR, f"{name}.log")

        with open(log_file, "w") as lf:
            process = subprocess.run(cmd, env=env, stdout=lf, stderr=lf)

        if process.returncode != 0:
            print(f"[GPU {gpu_id}] Task {name} FAILED (Return Code: {process.returncode})")
        else:
            print(f"[GPU {gpu_id}] Finished: {name}")


def main():
    if os.name == 'posix':
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))

    task_queue = multiprocessing.Queue()

    for task in TASKS:
        task_queue.put(task)

    for _ in GPUS:
        task_queue.put(None)

    processes = []
    for gpu_id in GPUS:
        p = multiprocessing.Process(target=worker_process, args=(gpu_id, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All tasks finished!")


if __name__ == "__main__":
    main()
