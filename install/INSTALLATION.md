# Installation

## 1. Clone and create the environment

```bash
git clone https://github.com/OpenGVLab/RSI-Nav.git
cd RSI-Nav
export RSI_NAV_ROOT="$PWD"
conda create -n rsi-nav python=3.11.12 -y
conda activate rsi-nav
export PIP_CONSTRAINT="$RSI_NAV_ROOT/install/requirements.txt"
```

Use Linux with an NVIDIA GPU, a compatible driver, and the CUDA 12.8 toolkit (`nvcc`). Headless rendering requires **OpenGL/EGL**, not Vulkan ([Habitat-Sim build guide](https://github.com/facebookresearch/habitat-sim/blob/v0.3.3/BUILD_FROM_SOURCE.md)).

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git wget pkg-config \
  libjpeg-dev libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev \
  libglew-dev libopencv-dev libegl1-mesa-dev \
  ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev \
  libavfilter-dev libswscale-dev libswresample-dev libpostproc-dev

python -m pip install pip==25.0.1 setuptools==75.8.2 wheel==0.45.1 \
  numpy==1.26.4 cmake==4.0.0 ninja==1.11.1.4 packaging==24.2
python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

## 2. Install Habitat 0.3.3

```bash
cd "$RSI_NAV_ROOT"
git clone --branch v0.3.3 --recursive https://github.com/facebookresearch/habitat-sim.git
git clone --branch v3.1.9 https://github.com/AcademySoftwareFoundation/Imath.git
python scripts/patch_simulator_sources.py habitat-sim "$RSI_NAV_ROOT/habitat-sim"

cd habitat-sim
python setup.py install --headless --with-cuda \
  --cmake-args="-DFETCHCONTENT_SOURCE_DIR_IMATH=$RSI_NAV_ROOT/Imath"

cd "$RSI_NAV_ROOT"
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
python -m pip install -e habitat-lab -e habitat-baselines

cd "$RSI_NAV_ROOT"
python -m pip install -r "$RSI_NAV_ROOT/install/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

## 3. Install native kernels

Place `flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl` in the repository root.

```bash
cd "$RSI_NAV_ROOT"
python -m pip install ./flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
wget -nc 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl'
python -m pip install ./causal_conv1d-1.6.0+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

## 4. Install MatterSim for DE→CE conversion

```bash
cd "$RSI_NAV_ROOT"
git clone --recursive https://github.com/peteanderson80/Matterport3DSimulator.git
cd Matterport3DSimulator
git -C pybind11 fetch --tags
git -C pybind11 checkout v2.11.1
python "$RSI_NAV_ROOT/scripts/patch_simulator_sources.py" mattersim "$RSI_NAV_ROOT/Matterport3DSimulator"
cmake -S . -B build -DEGL_RENDERING=ON \
  -DPYTHON_EXECUTABLE="$(command -v python)"
cmake --build build -j8
cd "$RSI_NAV_ROOT"
```

## 5. Set runtime paths

Run after activating the environment; set `RSI_NAV_ROOT` to your checkout path in a new shell.

```bash
export PYTHONPATH="$RSI_NAV_ROOT:$RSI_NAV_ROOT/eval/Nav:$RSI_NAV_ROOT/Matterport3DSimulator/build:${PYTHONPATH:-}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export TOKENIZERS_PARALLELISM=false
```

See [README.md](../README.md) for data preparation and usage.
