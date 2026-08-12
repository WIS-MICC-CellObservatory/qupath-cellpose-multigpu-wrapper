# Multi-GPU Cellpose-SAM Wrapper

A drop-in replacement for the python executable that QuPath's [BIOP Cellpose
extension](https://github.com/BIOP/qupath-extension-cellpose) calls, which
fans tile-based Cellpose-SAM inference out across multiple GPUs (or multiple
concurrent processes on a single GPU, or across CPU cores with no GPU at
all). Also usable completely standalone, outside QuPath, on any folder of
images.

## Why this exists

QuPath's StarDist extension runs natively inside QuPath's own JVM, so it can
multi-thread tile inference in-process. Cellpose-SAM doesn't run natively —
the extension exports annotation tiles to a temp folder and shells out to a
single external `python -m cellpose --dir <folder> ...` call, which loads the
model once and then processes every tile in that folder **sequentially**.
On a machine with more than one GPU, that leaves the rest of them idle.

This script sits in between: it intercepts that one call, splits the tiles
across N real cellpose processes (one or more per physical GPU, or across
CPU cores if there's no GPU), waits for all of them, and stitches the mask
outputs back into the original folder — so from QuPath's point of view,
nothing changed except it's faster.

## Contents

| File | Purpose |
|---|---|
| `multi_gpu_cellpose_wrapper.py` | The wrapper itself |
| `wrapper_config.example.json` | Template for the per-workstation config (see [Configuration](#configuration)) |
| `wrapper.bat` | Optional launcher, only needed if a preference field won't accept a bare `.py` |
| `.gitignore` | Excludes build artifacts and any real (machine-specific) config from version control |

## How it works

1. Reads the exact command-line arguments it was called with (whatever
   QuPath — or you — passed in), untouched.
2. Finds the `--dir <folder>` argument and lists the image tiles inside
   (`.tif` / `.tiff` / `.png` / `.jpg` / `.jpeg`), skipping anything that
   looks like a leftover output from a previous run.
3. Splits those tiles round-robin across however many worker processes your
   config specifies.
4. Copies (or symlinks) each worker's subset into its own scratch
   subfolder, launches a real cellpose process per worker with
   `CUDA_VISIBLE_DEVICES` set to that worker's assigned GPU (and CPU thread
   limits set, if enabled), and waits for all of them.
5. Moves every output file back into the original folder, so the caller
   finds results exactly where it expects them.

## Requirements

- A working Cellpose-SAM (`cpsam`) environment — this wrapper calls into
  your existing install, it doesn't bundle one.
- Windows (paths and the PyInstaller build step below assume this; the
  config-location fallback logic works on Linux/macOS too, but hasn't been
  tested end-to-end there).

## Setup with QuPath

1. **Confirm the real invocation shape first.** With your real python still
   configured in the extension, run one detection and check QuPath's log
   (`View > Show log`) for the exact command line it prints. This wrapper
   assumes something like `-m cellpose --dir <folder> ...`, optionally
   preceded by interpreter flags like `-Xutf8 -W ignore` — if your installed
   extension version constructs something meaningfully different, the
   argument handling in step 2 below may need adjusting.

2. **Build the executable** (see [Building the .exe](#building-the-exe) —
   most Cellpose preference fields want an actual `.exe`, not a bare `.py`).

3. **Locate the preference field.** In recent versions this has been a
   single field named something like *"Cellpose-SAM python.exe location"*
   rather than a separate venv/conda environment picker. Point it at the
   built `.exe`.

4. **First run generates a config.** Run one detection. With no config yet
   present for this machine, the wrapper writes a default one and the
   detection will fail with a message in the log telling you where it wrote
   it and to edit it. See [Configuration](#configuration) for what to put
   there, then rerun.

### Sharing one copy of the .exe across multiple workstations

The `.exe` can genuinely be a single shared copy — e.g. on a network path
every workstation's QuPath install points at — while each workstation keeps
its **own** settings (GPU count, path to its own local python environment).
That works because the config file does **not** live next to the `.exe`;
it lives at a fixed per-machine location:

```
%PROGRAMDATA%\CellposeWrapper\wrapper_config.json
```

`PROGRAMDATA` is used deliberately instead of `LOCALAPPDATA`: it's tied to
the physical machine, whereas `LOCALAPPDATA` follows the logged-in *user*
and can roam across different workstations on networks with roaming
profiles — which would silently apply the wrong machine's GPU settings.

Each workstation gets its own config generated on first run (see step 4
above) at that same fixed path, all pointing at the one shared `.exe`.

If a workstation's account can't write under `C:\ProgramData\` (some IT
policies restrict this), set the environment variable
`CELLPOSE_WRAPPER_CONFIG_DIR` on that machine to point at a folder it can
write to instead.

## Standalone use (outside QuPath)

Invoke it exactly like you'd invoke real cellpose through python — just
swap `python` for this wrapper:

```bat
multi_gpu_cellpose_wrapper.exe -m cellpose --dir "C:\path\to\folder" --pretrained_model cpsam --chan 0 --diameter 0 --save_tif --no_npy --use_gpu --verbose
```

It reads the same per-workstation config described above. Files in the
target folder that aren't recognized image extensions are left completely
untouched, and files that look like outputs from a previous run are
skipped rather than re-processed as new inputs — so it's safe to rerun on
the same folder repeatedly.

**Caveats:**
- File discovery is **non-recursive** — only files directly inside the
  given `--dir` are considered, not subfolders. If you pass cellpose's own
  `--look_one_level_down` flag, this wrapper won't mirror that when
  deciding what to split across workers, so results would differ from a
  plain (non-wrapped) cellpose call in that specific case.
- If you're used to running this via QuPath with a Groovy script that does
  channel subsetting / percentile normalization *before* export, that
  preprocessing happens on QuPath's side — a folder of raw, unprocessed
  tiffs given to the wrapper standalone skips that step entirely, so expect
  different segmentation behavior unless you've replicated it separately.

## Configuration

The wrapper reads `wrapper_config.json` from the per-machine location
described above (see `wrapper_config.example.json` in this repo for a
template — copy it there and edit it, don't just rename it in place, since
the real file's path and contents are machine-specific and intentionally
excluded from version control).

| Key | Meaning |
|---|---|
| `real_python` | Full path to the python executable inside your actual cellpose environment |
| `gpu_ids` | Which physical GPU indices to use (`nvidia-smi -L` lists them). On a CPU-only machine, set this to a single placeholder, e.g. `[0]` |
| `workers_per_gpu` | How many concurrent cellpose processes to run *per* GPU listed above. `1` = one process per GPU. Higher values run multiple processes on the same GPU — useful when a single process doesn't saturate it, at the cost of each process loading its own full copy of the model into VRAM |
| `copy_instead_of_symlink` | `true` copies tiles into each worker's scratch folder (safe everywhere, including Windows without Developer Mode); `false` symlinks instead (faster, avoids duplicating large files, but needs symlink permissions) |
| `limit_threads_per_worker` | Divides CPU cores evenly across concurrent workers (`OMP_NUM_THREADS` / `MKL_NUM_THREADS` / etc.) instead of letting each one try to grab every core. Matters most with no GPU; harmless to leave on otherwise |

### Tuning `workers_per_gpu`

Start at `1` per GPU, or `2` if you specifically want to test whether a
single process is leaving a GPU underutilized. While a run is in progress,
watch `nvidia-smi`:

- **GPU-Util%** climbing higher than a single process gave you → real gain.
- **Memory used** approaching the card's limit → back off. Each additional
  worker on the same GPU loads its own complete copy of the Cellpose-SAM
  model (a notably large network compared to earlier cellpose versions),
  so headroom disappears faster than you might expect. Going too high
  causes a worker to hit CUDA out-of-memory, which fails that worker's
  subprocess and, downstream, causes QuPath to report a generic
  `NullPointerException` on a missing mask file — not an obviously
  GPU-related message, so if you see that error, VRAM pressure from
  `workers_per_gpu` being set too high is one of the first things worth
  ruling out.

On a **shared multi-user machine**, also check `nvidia-smi` for other
users' processes on a given GPU before assuming a failure is this wrapper's
fault — a GPU already loaded up by someone else's job will fail first under
the same VRAM math.

### CPU-only machines

Still works — setting `CUDA_VISIBLE_DEVICES` on a machine with no CUDA
device is a harmless no-op. Two adjustments:
- Remove `--use_gpu` from whichever cellpose arguments you're passing in
  (that's a builder/CLI setting, not something this wrapper controls).
- Set `gpu_ids` to a single placeholder (e.g. `[0]`) and use
  `workers_per_gpu` to control how many concurrent CPU processes run.

Unlike the GPU case, there's no equally clean single metric (like
GPU-Util%) to watch — CPU workers share memory bandwidth and cache in ways
that are harder to predict, so benchmark actual wall-clock time at a couple
of different `workers_per_gpu` values rather than assuming more is better.

## Building the .exe

PyInstaller does **not** cross-compile — build on the actual Windows
machine that will run it:

```bat
pip install pyinstaller
python -m pyinstaller --onefile multi_gpu_cellpose_wrapper.py
```

(`python -m PyInstaller ...` is the more reliable invocation if the bare
`pyinstaller` command isn't found — it's usually a `PATH` issue with pip's
`Scripts` folder.)

**Known gotcha:** if you have an old standalone `pathlib` package installed
(a leftover backport from before pathlib was part of the standard library),
PyInstaller refuses to run alongside it with an explicit error naming the
package. Remove it first:
```bat
conda remove pathlib
REM or, if it was installed via pip into the env instead:
pip uninstall pathlib
```

This produces `dist\multi_gpu_cellpose_wrapper.exe`. Test it standalone
against a folder of a few dummy tiles before pointing QuPath at it — confirm
the `[wrapper] ...` diagnostic lines look sane and masks actually appear
back in the folder.

## Known issues / things to watch for

- **Interpreter flags before `-m cellpose`.** The BIOP extension may prepend
  flags like `-Xutf8 -W ignore` before `-m cellpose` in the call it makes.
  This wrapper handles that by copying the *entire* original argument list
  and only swapping the value that follows `--dir`, rather than trying to
  reconstruct the command from parts — this keeps it robust to whatever
  extra flags a given extension version adds, without needing to know about
  them in advance.
- **`OUTPUT_NAME_TAGS`** (the filename patterns used to recognize and skip
  leftover mask/flow files from a previous run) is based on common cellpose
  `--save_tif` naming conventions and may not be exhaustive for every
  cellpose version. If you notice old output files being reprocessed as new
  input tiles, check their actual suffix and add it to that list in the
  script.
- Extension internals (exact invocation shape, preference field naming,
  whether it wraps calls through conda activation vs. a direct executable
  path) can differ across `qupath-extension-cellpose` versions. If
  something here doesn't match what you see in QuPath's log, that's the
  extension version diverging from what this was built against, not
  necessarily a bug in this script — check the log first.

## License

MIT — see below, or replace with whatever your institution/lab prefers.
