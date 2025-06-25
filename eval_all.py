import os
import subprocess
import multiprocessing
import sys

import torch

GPUS = [i for i in range(torch.cuda.device_count())]

TASKS = []

LOGS_DIR = "logs"

names = ["11", "15"]
for name in names:
    TASKS.append(
        (f"MARS_{name}_ref", f"data/MARS/{name}", "Colmap", f"{LOGS_DIR}/MARS/{name}.pth", f"output/MARS/{name}", ""))

names = ["37", "41"]
for name in names:
    TASKS.append((f"MARS_{name}_ref", f"data/MARS/{name}", "Colmap", f"{LOGS_DIR}/MARS/{name}.pth", f"output/MARS/{name}",
                  "--rap_resolution 4"))

TASKS.append(("new_church_ref", "data/new_church", "Colmap", f"{LOGS_DIR}/new_church.pth", "output/new_church", ""))
TASKS.append(("aachen_ref", "data/aachen_sub", "Colmap", f"{LOGS_DIR}/aachen.pth", "output/aachen", ""))

names = ["shop", "hospital", "college", "church"]
for name in names:
    TASKS.append((f"Cam_{name}_ref", f"data/Cambridge/{name}", "Cambridge", f"{LOGS_DIR}/Cambridge/{name}.pth",
                  f"output/Cambridge/{name}", ""))

TASKS.append((f"Cam_court_ref", f"data/Cambridge/court/colmap/undistorted", "Colmap", f"{LOGS_DIR}/Cambridge/court.pth",
              f"output/Cambridge/court", ""))

names = ["chess", "fire", "heads", "office", "pumpkin", "kitchen", "stairs"]
for name in names:
    TASKS.append((f"7Scn_{name}_ref", f"data/7Scenes/{name}", "7Scenes", f"{LOGS_DIR}/7Scenes/{name}.pth",
                  f"output/7Scenes/{name}", ""))
for name in names:
    TASKS.append((f"7Scn_{name}_sfm_ref", f"data/7Scenes_sfm/{name}", "Colmap", f"{LOGS_DIR}/7Scenes_sfm/{name}.pth",
                  f"output/7Scenes_sfm/{name}", ""))

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

        name, data_dir, tag, ckpt_path, gs_dir, extra_args = task
        print(f"[GPU {gpu_id}] Starting: {name}")

        cmd = [
            sys.executable, "eval.py",
            "-n", name,
            "-d", data_dir,
            "-t", tag,
            "-p", ckpt_path,
            "-m", gs_dir
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
