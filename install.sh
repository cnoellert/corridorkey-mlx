#!/usr/bin/env bash
# install.sh — CorridorKey Flame PyBox installer
# Supports macOS (MLX/Apple Silicon) and Linux (CUDA/Rocky)
# Usage: bash install.sh [--weights /path/to/weights]
set -euo pipefail

INSTALL_ROOT="/opt/corridorkey"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="$(uname -s)"

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[corridorkey]${NC} $*"; }
warn()  { echo -e "${YELLOW}[corridorkey]${NC} $*"; }
error() { echo -e "${RED}[corridorkey] ERROR${NC} $*"; exit 1; }

# ── Args ───────────────────────────────────────────────────────────────────
WEIGHTS_SRC=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --weights) WEIGHTS_SRC="$2"; shift 2 ;;
        *) error "Unknown argument: $1" ;;
    esac
done

# ── Platform ───────────────────────────────────────────────────────────────
if [[ "$PLATFORM" == "Darwin" ]]; then
    CONDA_ENV="corridorkey-mlx"
    WEIGHTS_FILE="CorridorKey_v1.0.mlx.npz"
    info "Platform: macOS (MLX)"
elif [[ "$PLATFORM" == "Linux" ]]; then
    CONDA_ENV="corridorkey-cuda"
    WEIGHTS_FILE="CorridorKey_v1.0.pth"
    info "Platform: Linux (CUDA)"
else
    error "Unsupported platform: $PLATFORM"
fi

# ── Conda ──────────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo ""
    echo -e "${RED}[corridorkey] ERROR${NC} conda not found."
    echo ""
    echo "  Install Miniconda first, then re-run this script:"
    echo ""
    if [[ "$PLATFORM" == "Darwin" ]]; then
        echo "    brew install miniconda"
        echo "    # or: https://docs.conda.io/en/latest/miniconda.html"
    else
        echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "    bash Miniconda3-latest-Linux-x86_64.sh"
        echo "    # then restart your shell"
    fi
    echo ""
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -q "^$CONDA_ENV "; then
    info "Conda env '$CONDA_ENV' already exists — skipping creation"
else
    info "Creating conda env: $CONDA_ENV (Python 3.11)"
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

conda activate "$CONDA_ENV"

# ── Python deps ────────────────────────────────────────────────────────────
info "Installing Python dependencies..."
if [[ "$PLATFORM" == "Darwin" ]]; then
    pip install --quiet mlx numpy openexr opencv-python-headless
else
    # Detect CUDA version for correct PyTorch wheel
    if command -v nvidia-smi &>/dev/null; then
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | cut -d. -f1,2 | tr -d .)
        info "Detected CUDA $CUDA_VER"
        pip install --quiet torch torchvision \
            --index-url "https://download.pytorch.org/whl/cu${CUDA_VER}"
    else
        warn "nvidia-smi not found — installing CPU-only PyTorch"
        pip install --quiet torch torchvision
    fi
    pip install --quiet numpy openexr opencv-python-headless timm scipy
fi
info "Dependencies installed"

# ── Directory structure ────────────────────────────────────────────────────
info "Creating /opt/corridorkey/..."
sudo mkdir -p "$INSTALL_ROOT"/{models,pybox,reference}
sudo chown -R "$(whoami)" "$INSTALL_ROOT"

# ── PyBox files ────────────────────────────────────────────────────────────
info "Installing pybox files..."
cp "$REPO_DIR"/pybox/corridorkey_pybox.py      "$INSTALL_ROOT/pybox/"
cp "$REPO_DIR"/pybox/corridorkey_daemon_mlx.py "$INSTALL_ROOT/pybox/"
cp "$REPO_DIR"/pybox/corridorkey_daemon_cuda.py "$INSTALL_ROOT/pybox/"

# ── Reference inference code (CUDA only) ──────────────────────────────────
if [[ "$PLATFORM" == "Linux" ]]; then
    info "Installing PyTorch reference inference code..."
    cp -r "$REPO_DIR/reference/CorridorKeyModule" "$INSTALL_ROOT/reference/"
    cp -r "$REPO_DIR/reference/utils"             "$INSTALL_ROOT/reference/"
fi

# ── Weights ────────────────────────────────────────────────────────────────
WEIGHTS_DEST="$INSTALL_ROOT/models/$WEIGHTS_FILE"
WEIGHTS_URL="https://github.com/cnoellert/corridorkey-flame/releases/download/v1.0.0/$WEIGHTS_FILE"

if [[ -f "$WEIGHTS_DEST" ]]; then
    info "Weights already present: $WEIGHTS_DEST"
elif [[ -n "$WEIGHTS_SRC" ]]; then
    if [[ -f "$WEIGHTS_SRC" ]]; then
        info "Copying weights from $WEIGHTS_SRC"
        cp "$WEIGHTS_SRC" "$WEIGHTS_DEST"
    else
        error "Weights file not found: $WEIGHTS_SRC"
    fi
else
    info "Downloading weights (~380MB)..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$WEIGHTS_DEST" "$WEIGHTS_URL" || error "Download failed: $WEIGHTS_URL"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "$WEIGHTS_DEST" "$WEIGHTS_URL" || error "Download failed: $WEIGHTS_URL"
    else
        error "Neither curl nor wget found. Install one or pass --weights /path/to/$WEIGHTS_FILE"
    fi
    info "Weights downloaded: $WEIGHTS_DEST"
fi

# ── Flame path ─────────────────────────────────────────────────────────────
PYBOX_HANDLER="$INSTALL_ROOT/pybox/corridorkey_pybox.py"
echo ""
info "Installation complete."
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  In Flame: Batch > Add Node > PyBox"
echo -e "  Point at: ${YELLOW}$PYBOX_HANDLER${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
