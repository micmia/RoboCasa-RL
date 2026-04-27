#!/bin/bash
# ============================================================================
# _common.sh — Shared SLURM environment setup for all NAV4RAIL jobs
# ============================================================================
# Source this file from batch scripts (do not use dirname(BASH_SOURCE) alone: Slurm runs a
# copy under /var/spool/slurmd/.../slurm_script, so BASH_SOURCE would miss this file):
#   if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -r "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh" ]; then
#     source "${SLURM_SUBMIT_DIR}/scripts/slurm/_common.sh"
#   else
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
#   fi
#
# What it does:
#   1. cd to benchmarking/ root (reliable cwd regardless of sbatch location)
#   2. Load CUDA module (for GPU access)
#   3. Create venv if missing, activate it
#   4. Load the same Python module as the venv base interpreter (LD_LIBRARY_PATH for
#      libbz2 etc.) and clear PYTHONHOME (module load sets it, breaks venvs)
#   5. Ensure all requirements.txt deps are installed (idempotent, flock-serialized)
#   6. Set PYTHONPATH so `python -m src.*` works
#   7. Print diagnostic info to stdout (captured by SLURM .out)
# ============================================================================

set -euo pipefail

# ── Resolve benchmarking/ root ──────────────────────────────────────────────
# Works whether this file is sourced from scripts/slurm/ or anywhere else.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_DIR"

# ── Auto-load .env (HF_TOKEN, WANDB_API_KEY, VAST_API_KEY, …) ───────────────
# Rsynced from the workstation by `cluster_exec.sh submit`. Any already-exported
# variable wins (Slurm env > .env), so users can override via `VAR=val sbatch`.
if [ -r "$BENCH_DIR/.env" ]; then
    echo "[_common.sh] Loading env from $BENCH_DIR/.env"
    set -a
    # shellcheck disable=SC1091
    . "$BENCH_DIR/.env"
    set +a
fi

VENV_DIR="${VENV_DIR:-$HOME/.venvs/nav4rail_bench}"
# Partition-scoped venv: P100 and 3090 need different package builds (torch, bitsandbytes).
if [ -n "${SLURM_JOB_PARTITION:-}" ]; then
    VENV_DIR="${VENV_DIR}_${SLURM_JOB_PARTITION}"
fi

# ── Load modules ────────────────────────────────────────────────────────────
# CUDA is needed for GPU. We load python module ONLY for venv creation (the
# compute node's system Python may be too old), then let the venv's own Python
# take over.
module load cuda/12.4.1 2>/dev/null || true

# ── Create venv if it doesn't exist ────────────────────────────────────────
# Check for bin/activate (not just the directory): python3 -m venv creates
# the directory BEFORE populating it, so concurrent tasks that see the dir
# but source a not-yet-created activate script get "No such file or directory".
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    (
        flock -x 200
        if [ ! -f "$VENV_DIR/bin/activate" ]; then
            echo "[_common.sh] Creating venv at $VENV_DIR ..."
            # Load python module for venv creation (need >= 3.10)
            module load python/3.11.13 2>/dev/null || true
            python3 -m venv "$VENV_DIR"
        fi
    ) 200>>"${VENV_DIR}.lock"
fi

# ── Activate venv ───────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── CRITICAL FIX: Clear PYTHONHOME ──────────────────────────────────────────
# `module load python/X.Y.Z` often sets PYTHONHOME, which overrides the venv's
# sys.path discovery. This makes Python ignore the venv's site-packages and
# look for packages under the module's prefix instead.
# Result: "ModuleNotFoundError: No module named 'yaml'" even though pyyaml is
# installed in the venv.
# See: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHOME
unset PYTHONHOME 2>/dev/null || true

# ── Python module runtime libs (libbz2 → _bz2 → datasets import chain) ───────
# Venv creation loads python/3.11.13 once; batch jobs otherwise skip it, and
# stripped compute images may not ship libbz2 on the default linker path.
# HPC Python modules typically bundle libbz2 in their own lib/ dir (next to
# the interpreter); we add that to LD_LIBRARY_PATH with fallbacks to system
# paths, `module load bzip2`, and finally `ldconfig -p`.
#
# CPython 3.11 from python.org / HPC stacks links `_bz2` against the exact
# SONAME `libbz2.so.1.0` (legacy). Modern distros (Ubuntu 22.04+) ship only
# `libbz2.so.1.0.<X>` under a `libbz2.so.1` symlink. Adding the directory to
# LD_LIBRARY_PATH is NOT enough — the loader still fails on the missing
# `.1.0` alias. We therefore materialise a user-local shim into the venv's
# own lib/ so the interpreter's own linker sees the legacy SONAME.
module load python/3.11.13 2>/dev/null || true
unset PYTHONHOME 2>/dev/null || true

_SHIM_DIR="$VENV_DIR/lib/nav4rail_shims"
mkdir -p "$_SHIM_DIR"
case ":${LD_LIBRARY_PATH:-}:" in *":${_SHIM_DIR}:"*) ;; *)
    export LD_LIBRARY_PATH="${_SHIM_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
esac

_ensure_libbz2_alias() {
    # If `libbz2.so.1.0` is missing from $1 but `libbz2.so.1` (or libbz2.so.1.0.X)
    # is present, drop a symlink into $_SHIM_DIR pointing at the real file.
    local dir="$1" real=""
    [ -d "$dir" ] || return 1
    if [ -f "$dir/libbz2.so.1.0" ]; then
        real="$dir/libbz2.so.1.0"
    elif [ -f "$dir/libbz2.so.1" ]; then
        real="$dir/libbz2.so.1"
    else
        real=$(ls "$dir"/libbz2.so.1.0.* 2>/dev/null | head -1 || true)
        [ -z "$real" ] && real=$(ls "$dir"/libbz2.so.* 2>/dev/null | head -1 || true)
    fi
    [ -z "$real" ] && return 1
    if [ ! -e "$_SHIM_DIR/libbz2.so.1.0" ]; then
        ln -sf "$real" "$_SHIM_DIR/libbz2.so.1.0"
    fi
    if [ ! -e "$_SHIM_DIR/libbz2.so.1" ]; then
        ln -sf "$real" "$_SHIM_DIR/libbz2.so.1"
    fi
    return 0
}

_try_libbz2_dir() {
    [ -z "${1:-}" ] && return 1
    if [ -f "$1/libbz2.so.1.0" ] || [ -f "$1/libbz2.so.1" ] \
       || ls "$1"/libbz2.so.1.0.* >/dev/null 2>&1; then
        case ":${LD_LIBRARY_PATH:-}:" in *":${1}:"*) ;; *)
            export LD_LIBRARY_PATH="${1}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
        esac
        _ensure_libbz2_alias "$1" || true
        return 0
    fi
    return 1
}

_libbz2_done=""

# 1) Python module's own lib dir (base_prefix): HPC builds usually bundle libbz2 here.
_py_base_prefix=$(python -c 'import sys; print(sys.base_prefix)' 2>/dev/null || true)
if [ -n "$_py_base_prefix" ]; then
    for _bz_dir in "$_py_base_prefix/lib" "$_py_base_prefix/lib64"; do
        if _try_libbz2_dir "$_bz_dir"; then _libbz2_done=1; break; fi
    done
fi

# 2) Standard system library paths.
if [ -z "$_libbz2_done" ]; then
    for _bz_dir in /usr/lib/x86_64-linux-gnu /usr/lib64 /lib64 /usr/lib; do
        if _try_libbz2_dir "$_bz_dir"; then _libbz2_done=1; break; fi
    done
fi

# 3) Explicit bzip2 environment module.
if [ -z "$_libbz2_done" ]; then
    if module load bzip2 2>/dev/null; then
        # Rescan common locations after the module appended its own prefix.
        for _bz_dir in $(echo "${LD_LIBRARY_PATH:-}" | tr ':' ' '); do
            _try_libbz2_dir "$_bz_dir" && _libbz2_done=1 && break
        done
    fi
fi

# 4) ldconfig cache (needs a path writable by the admin at image-build time).
if [ -z "$_libbz2_done" ] && command -v ldconfig >/dev/null 2>&1; then
    _bz_lib=$(ldconfig -p 2>/dev/null | awk '/libbz2\.so\.1/ { print $NF; exit }' || true)
    if [ -n "$_bz_lib" ]; then
        _bz_dir=$(dirname "$_bz_lib")
        _try_libbz2_dir "$_bz_dir" && _libbz2_done=1
    fi
    unset _bz_lib
fi

# Final sanity probe: if the shim directory has an alias but `python -c "import bz2"`
# still fails, surface a clearer error than "ImportError: libbz2.so.1.0" in job logs.
if ! python -c "import bz2" 2>/dev/null; then
    echo "[_common.sh] WARNING: 'python -c \"import bz2\"' failed."
    echo "  Tried:"
    echo "    - ${_py_base_prefix:-<no base_prefix>}/lib{,64}"
    echo "    - /usr/lib/x86_64-linux-gnu, /usr/lib64, /lib64, /usr/lib"
    echo "    - 'module load bzip2'"
    echo "    - 'ldconfig -p'"
    echo "  Shim dir: $_SHIM_DIR (contents:)"
    ls -la "$_SHIM_DIR" 2>/dev/null || true
    echo "  Effective LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}"
fi
unset _bz_dir _libbz2_done _py_base_prefix _SHIM_DIR
unset -f _try_libbz2_dir _ensure_libbz2_alias

# ── Hugging Face token (needed for gated repos like google/gemma-2-9b-it) ─
if [ -z "${HF_TOKEN:-}" ]; then
    for _hf_token_file in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
        if [ -f "$_hf_token_file" ]; then
            export HF_TOKEN="$(cat "$_hf_token_file")"
            break
        fi
    done
    unset _hf_token_file
fi
if [ -n "${HF_TOKEN:-}" ]; then
    export HF_TOKEN                         # ensure child Python sees it
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"  # newer HF_HUB clients look here
else
    echo "[_common.sh] WARNING: HF_TOKEN not set — gated models (e.g. gemma) will fail."
    echo "  Fix: add HF_TOKEN=hf_... to $BENCH_DIR/.env, or huggingface-cli login."
fi

# ── Weights & Biases key (track training AND inference runs) ────────────────
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.netrc" ]; then
    # `wandb login` writes the key to ~/.netrc under machine api.wandb.ai
    _wandb_key=$(awk '/machine api.wandb.ai/{f=1} f && /password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null || true)
    if [ -n "$_wandb_key" ]; then
        export WANDB_API_KEY="$_wandb_key"
    fi
    unset _wandb_key
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY                    # ensure child Python sees it
else
    echo "[_common.sh] WARNING: WANDB_API_KEY not set — W&B tracking will be disabled."
    echo "  Fix: add WANDB_API_KEY=... to $BENCH_DIR/.env, or wandb login."
fi
export WANDB_PROJECT="${WANDB_PROJECT:-nav4rail-bench}"

# ── Ensure all dependencies are installed ───────────────────────────────────
# Serialize pip against VENV_DIR.lock: a shared venv on NFS + concurrent array
# tasks causes errno 116 (Stale file handle) and half-installed envs → import yaml fails.
# This is idempotent: pip skips satisfied deps; queued jobs wait on flock.
(
    flock -x 200
    # Avoid "invalid command 'bdist_wheel'" when building sdists from cache.
    python -m pip install --quiet --disable-pip-version-check wheel setuptools
    # ── GPU-aware PyTorch: P100 (sm_60) needs torch < 2.5 ──────────────────
    # Install the right version BEFORE requirements.txt so pip sees torch as
    # already satisfied and does not upgrade to an incompatible build.
    _gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$_gpu_cc" ]; then
        _cc_int=$(echo "$_gpu_cc" | tr -d '.')
        if [ "$_cc_int" -lt 75 ]; then
            echo "[_common.sh] GPU CC ${_gpu_cc} < 7.5 — pinning torch 2.4.x for sm_60 support"
            pip install --quiet --disable-pip-version-check "torch>=2.4.0,<2.5.0"
        fi
    fi
    pip install --quiet --disable-pip-version-check -r "$BENCH_DIR/requirements.txt" 2>&1 || {
        echo "[_common.sh] WARNING: pip install failed, retrying with --no-build-isolation..."
        pip install --quiet --disable-pip-version-check --no-build-isolation -r "$BENCH_DIR/requirements.txt"
    }
    # Cheap sanity check: repair if a previous crashed install left yaml missing.
    python -c "import yaml" 2>/dev/null || python -m pip install --quiet --disable-pip-version-check "pyyaml>=6.0"
) 200>>"$VENV_DIR.lock"

# ── PYTHONPATH ──────────────────────────────────────────────────────────────
# `module load python/X.Y.Z` prepends the module's own site-packages to
# PYTHONPATH (e.g. /projects/share/apps/python/3.11.13/lib/python3.11/
# site-packages). That silently SHADOWS the venv's site-packages — including
# the CC-pinned torch, pydantic, transformers, etc. — because entries on
# PYTHONPATH take precedence over the venv's pyvenv.cfg discovery.
# Strip any PYTHONPATH entry rooted at sys.base_prefix before appending our
# project dir, so the venv wins.
if [ -n "${PYTHONPATH:-}" ]; then
    _py_base_prefix=$(python -c 'import sys; print(sys.base_prefix)' 2>/dev/null || true)
    if [ -n "$_py_base_prefix" ]; then
        _clean_pp=""
        IFS=':' read -ra _pp_arr <<< "$PYTHONPATH"
        for _entry in "${_pp_arr[@]}"; do
            case "$_entry" in
                "$_py_base_prefix"|"$_py_base_prefix"/*) continue ;;
            esac
            _clean_pp="${_clean_pp:+$_clean_pp:}$_entry"
        done
        export PYTHONPATH="$_clean_pp"
        unset _clean_pp _pp_arr _entry
    fi
    unset _py_base_prefix
fi
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}${BENCH_DIR}"

# ── Diagnostic output (captured in SLURM .out) ─────────────────────────────
echo "=== NAV4RAIL Environment ==="
echo "  Date:       $(date)"
echo "  Hostname:   $(hostname)"
echo "  Working dir: $(pwd)"
echo "  Python:     $(which python) ($(python --version 2>&1))"
echo "  VENV_DIR:   $VENV_DIR"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  PYTHONHOME: ${PYTHONHOME:-<unset>}"
echo "  GPU:        $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  yaml check: $(python -c 'import yaml; print(yaml.__version__)' 2>&1)"
echo "  bz2 check:  $(python -c "import bz2; print('ok')" 2>&1)"
echo "============================="
