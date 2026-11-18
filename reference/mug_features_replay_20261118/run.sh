#!/usr/bin/env bash
set -euo pipefail
RUN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$RUN")
REPO="$ROOT/vendor/LIBERO-MAX"
export LD_LIBRARY_PATH="$ROOT/libero_max_pilot_20261115/system-libs/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MAGICK_HOME="$ROOT/libero_max_pilot_20261115/system-libs/root/usr"
export MAGICK_CODER_MODULE_PATH="$MAGICK_HOME/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders"
export MAGICK_CONFIGURE_PATH="$ROOT/libero_max_pilot_20261115/system-libs/root/etc/ImageMagick-6"
export HF_HOME="$ROOT/cache/huggingface" HF_HUB_OFFLINE=1 XDG_CACHE_HOME="$ROOT/cache"
export LIBERO_SOURCE_PACKAGE_ROOT="$REPO/.deps/LIBERO-plus/libero" LIBERO_CONFIG_PATH="$ROOT/libero_max_lite_20261115/plus-config"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec "$ROOT/.venv/bin/python" -u "$RUN/${EXPERIMENT_SCRIPT:-experiment.py}" "$@"
