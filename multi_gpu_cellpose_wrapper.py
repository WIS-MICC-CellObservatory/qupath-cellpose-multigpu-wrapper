#!/usr/bin/env python3
"""
multi_gpu_cellpose_wrapper.py  (v2 -- external config, exe-friendly)

Drop-in replacement for the "Cellpose-SAM python.exe location" that
QuPath's BIOP Cellpose extension calls. Intercepts the standard
invocation:

    <python> -m cellpose --dir <tiles_dir> [...other cellpose CLI args]

fans the tiles in <tiles_dir> out across multiple GPUs by launching one
real cellpose process per GPU (each processing its own subset of tiles
sequentially, but all GPUs running concurrently), waits for all of
them, then merges the mask outputs back into the original directory so
the extension finds them exactly where it expects.

------------------------------------------------------------------
CONFIG (shared .exe, per-workstation settings)
------------------------------------------------------------------
This script/exe is designed to be a single shared copy (e.g. on a
network path every QuPath install points at) while each workstation
keeps its OWN settings (GPU count/ids, path to that machine's real
python environment). Settings therefore do NOT live next to the exe
by default -- they live in a per-machine location:

    Windows:  %PROGRAMDATA%\CellposeWrapper\wrapper_config.json
              (e.g. C:\ProgramData\CellposeWrapper\wrapper_config.json)

PROGRAMDATA is deliberately used instead of LOCALAPPDATA: PROGRAMDATA
is tied to the physical machine, whereas LOCALAPPDATA follows the
logged-in user profile and can roam across different workstations on
networks using roaming profiles -- which would silently apply GPU
settings from the wrong machine.

On first run on a given workstation (with no existing config), this
script creates a default file at that per-machine path and exits with
a message -- edit real_python / gpu_ids there for that machine, then
rerun. Every workstation ends up with its own file at the same fixed
path, all pointing at the one shared .exe.

Override (optional, e.g. for testing multiple configs from one
machine): set an environment variable CELLPOSE_WRAPPER_CONFIG_DIR to
a folder, and the config file is read/written there instead.

If PROGRAMDATA isn't set (e.g. running on Linux/macOS during testing),
this falls back to a folder next to wherever the script/exe itself is
running from.
------------------------------------------------------------------

Expected wrapper_config.json shape:
{
  "real_python": "C:\\Users\\you\\miniconda3\\envs\\cellpose\\python.exe",
  "gpu_ids": [0, 1, 2, 3],
  "workers_per_gpu": 1,
  "copy_instead_of_symlink": true,
  "limit_threads_per_worker": true
}

"limit_threads_per_worker" divides available CPU cores evenly across
however many worker processes are launched (via OMP_NUM_THREADS /
MKL_NUM_THREADS / OPENBLAS_NUM_THREADS / NUMEXPR_NUM_THREADS), rather
than letting every worker try to grab all cores. Matters most in a
CPU-only setup (see below); harmless to leave on with a GPU too, since
it only affects CPU-side thread pools.

CPU-ONLY MACHINES (no GPU at all): this still works, since setting
CUDA_VISIBLE_DEVICES on a machine with no CUDA device is a harmless
no-op. Two adjustments for that case:
  - Remove "--use_gpu" from the cellpose arguments you're passing in
    (that flag lives in your Cellpose2D builder config / CLI call, not
    in this wrapper -- it just passes args through unchanged).
  - "gpu_ids" becomes a bit of a misnomer here -- just set it to a
    single placeholder value, e.g. [0], and use "workers_per_gpu" to
    control how many concurrent CPU processes run, e.g. gpu_ids=[0],
    workers_per_gpu=4 for 4 concurrent CPU worker processes.

"workers_per_gpu" controls how many concurrent cellpose processes run
on EACH physical GPU listed in gpu_ids -- useful on a single GPU where
one cellpose process doesn't saturate it (small tiles, light model
relative to VRAM/compute available). E.g. gpu_ids=[0], workers_per_gpu=4
launches 4 concurrent processes all on GPU 0. Each process loads its
own full copy of the model into VRAM, so this trades VRAM headroom for
throughput -- start low (2) and watch `nvidia-smi` for both GPU
utilization% (should rise toward 100) and memory (should not near the
card's limit) before pushing higher. If a worker OOMs, its subprocess
returns nonzero and the whole detection call fails, so tune
conservatively rather than assuming more is always better.

------------------------------------------------------------------
STANDALONE USE (outside QuPath, run directly from a terminal)
------------------------------------------------------------------
Invoke it exactly the way you'd invoke real cellpose through python --
just swap "python" for this wrapper/exe:

    multi_gpu_cellpose_wrapper.exe -m cellpose --dir "C:\path\to\folder" ^
        --pretrained_model cpsam --chan 0 --diameter 0 --save_tif --no_npy --use_gpu --verbose

It reads GPU/worker settings from this workstation's config exactly
as it would when called by QuPath (see CONFIG section below). Files
in the target folder that aren't .tif/.tiff/.png/.jpg/.jpeg are left
completely untouched. Files that look like leftover outputs from a
previous run (see OUTPUT_NAME_TAGS below) are skipped rather than
re-processed as new tiles, so rerunning on the same folder is safe.

CAVEAT: file discovery is non-recursive -- only files directly inside
the given --dir are considered, not subfolders. If you pass cellpose's
own --look_one_level_down flag, cellpose itself would look one level
into subdirectories, but this wrapper currently won't mirror that when
deciding what to split across workers, so results would differ from a
plain (non-wrapped) cellpose call in that specific case.
------------------------------------------------------------------

------------------------------------------------------------------
BUILDING THE .exe (must be done ON WINDOWS -- PyInstaller does not
cross-compile from Linux/macOS)
------------------------------------------------------------------
    pip install pyinstaller
    pyinstaller --onefile multi_gpu_cellpose_wrapper.py

This produces dist\\multi_gpu_cellpose_wrapper.exe. Copy
wrapper_config.json into the SAME folder as that .exe (the script
looks for it next to the running executable, not next to the .py
source). Point QuPath's "Cellpose-SAM python.exe location" field at
the .exe.

------------------------------------------------------------------
IMPORTANT -- VERIFY BEFORE RELYING ON THIS
------------------------------------------------------------------
I can't confirm your installed extension version's exact invocation
shape. Before trusting this in a real analysis:
  a) With your real python still configured, run one detection and
     check QuPath's log (View > Show log) for the literal command
     line printed -- confirm it matches "-m cellpose --dir X ...".
  b) Test the built .exe standalone from a terminal first, against a
     folder of a few dummy tiles:
         multi_gpu_cellpose_wrapper.exe -m cellpose --dir C:\\temp\\testtiles --pretrained_model cpsam
     and confirm masks appear back in that folder.
  c) Only then point QuPath's preference field at the .exe.
"""

import sys
import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Filenames containing any of these are treated as previous cellpose
# OUTPUTS, not new input tiles -- protects against re-processing a
# folder that already has masks from an earlier run in it (QuPath's
# temp dirs are always fresh so this never matters there, but a
# standalone folder you reuse for repeated runs will accumulate these).
# This list is based on common cellpose CLI --save_tif naming and may
# not be exhaustive for every cellpose version -- if you spot leftover
# mask/flow files being reprocessed, check their exact suffix and add
# it here.
OUTPUT_NAME_TAGS = ("_cp_masks", "_masks", "_seg", "_flows", "_outlines", "_cellprob")

DEFAULT_CONFIG = {
    "real_python": "C:\\path\\to\\your\\real\\cellpose\\env\\python.exe",
    "gpu_ids": [0, 1, 2, 3],
    "workers_per_gpu": 1,
    "copy_instead_of_symlink": True,
    "limit_threads_per_worker": True,
}


def get_exe_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_config_dir():
    env_override = os.environ.get("CELLPOSE_WRAPPER_CONFIG_DIR")
    if env_override:
        return Path(env_override)
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        return Path(program_data) / "CellposeWrapper"
    # Fallback for non-Windows / standalone testing: next to the exe.
    return get_exe_dir()


def load_config():
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "wrapper_config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
        print(f"[wrapper] No config found for this workstation -- wrote "
              f"a default one to {config_path}. Edit real_python and "
              f"gpu_ids for THIS machine, then rerun.", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        cfg = json.load(f)
    for key in DEFAULT_CONFIG:
        cfg.setdefault(key, DEFAULT_CONFIG[key])
    return cfg


def split_round_robin(items, n_buckets):
    buckets = [[] for _ in range(n_buckets)]
    for i, item in enumerate(items):
        buckets[i % n_buckets].append(item)
    return buckets


def main():
    cfg = load_config()
    real_python = cfg["real_python"]
    gpu_ids = cfg["gpu_ids"]
    workers_per_gpu = cfg["workers_per_gpu"]
    copy_instead_of_symlink = cfg["copy_instead_of_symlink"]
    limit_threads_per_worker = cfg["limit_threads_per_worker"]

    # Expand e.g. gpu_ids=[0,1], workers_per_gpu=3 into
    # worker_gpu_list=[0,0,0,1,1,1] -- one entry per concurrent process,
    # each naming which physical GPU that process should use.
    worker_gpu_list = [g for g in gpu_ids for _ in range(workers_per_gpu)]

    argv = sys.argv[1:]

    if "--dir" not in argv:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_gpu_list[0])
        sys.exit(subprocess.call([real_python] + argv, env=env))

    dir_idx = argv.index("--dir")
    orig_dir = Path(argv[dir_idx + 1]).resolve()

    all_images = [
        p for p in orig_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    tile_files = sorted(
        p for p in all_images
        if not any(tag in p.stem for tag in OUTPUT_NAME_TAGS)
    )
    skipped = len(all_images) - len(tile_files)
    if skipped:
        print(f"[wrapper] skipped {skipped} file(s) in {orig_dir} that "
              f"look like previous cellpose outputs (name matched "
              f"{OUTPUT_NAME_TAGS}) -- not treating them as new tiles",
              file=sys.stderr)
    if not tile_files:
        print(f"[wrapper] No tile images found in {orig_dir}", file=sys.stderr)
        sys.exit(1)

    n_workers = min(len(worker_gpu_list), len(tile_files))
    worker_gpu_list = worker_gpu_list[:n_workers]
    buckets = split_round_robin(tile_files, n_workers)

    with tempfile.TemporaryDirectory(dir=orig_dir.parent) as scratch:
        scratch = Path(scratch)
        chunk_dirs = []
        chunk_input_names = {}

        for i in range(n_workers):
            chunk_dir = scratch / f"worker{i}_gpu{worker_gpu_list[i]}"
            chunk_dir.mkdir()
            chunk_dirs.append(chunk_dir)
            chunk_input_names[chunk_dir] = {f.name for f in buckets[i]}
            for f in buckets[i]:
                dest = chunk_dir / f.name
                if copy_instead_of_symlink:
                    shutil.copy2(f, dest)
                else:
                    os.symlink(f, dest)

        print(f"[wrapper] {len(tile_files)} tiles split across "
              f"{n_workers} worker(s) on GPU(s): {worker_gpu_list}",
              file=sys.stderr)

        cpu_count = os.cpu_count() or 1
        threads_per_worker = max(1, cpu_count // n_workers)
        if limit_threads_per_worker:
            print(f"[wrapper] limiting each worker to "
                  f"{threads_per_worker} CPU thread(s) "
                  f"({cpu_count} cores / {n_workers} workers) to avoid "
                  f"oversubscription -- matters most when running "
                  f"without a GPU, harmless otherwise", file=sys.stderr)

        procs = []
        for i, chunk_dir in enumerate(chunk_dirs):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(worker_gpu_list[i])
            if limit_threads_per_worker:
                for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[var] = str(threads_per_worker)
            # Copy the ORIGINAL argv untouched (preserves interpreter
            # flags like -Xutf8 / -W ignore in whatever position the
            # extension put them) and only swap the --dir value.
            worker_argv = list(argv)
            worker_argv[dir_idx + 1] = str(chunk_dir)
            cmd = [real_python] + worker_argv
            print(f"[wrapper] worker {i} (GPU {worker_gpu_list[i]}): "
                  f"{' '.join(cmd)}", file=sys.stderr)
            procs.append(subprocess.Popen(cmd, env=env))

        returncodes = [p.wait() for p in procs]

        if any(rc != 0 for rc in returncodes):
            print(f"[wrapper] one or more cellpose processes failed "
                  f"(worker, gpu_id, returncode): "
                  f"{list(zip(range(n_workers), worker_gpu_list, returncodes))}",
                  file=sys.stderr)
            sys.exit(1)

        moved = 0
        for chunk_dir in chunk_dirs:
            input_names = chunk_input_names[chunk_dir]
            for f in chunk_dir.iterdir():
                if f.name in input_names:
                    continue
                shutil.move(str(f), str(orig_dir / f.name))
                moved += 1

        print(f"[wrapper] merged {moved} output file(s) back into "
              f"{orig_dir}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
