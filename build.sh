#!/bin/bash
set -eo pipefail

# Define the build matrix
# PYTHON_VERSIONS=('3.9' '3.10' '3.11' '3.12' '3.13')
PYTHON_VERSIONS=('3.11')
TORCH_VERSIONS=('2.4.0' '2.5.1' '2.6.0' '2.7.0')
# CXX11_ABI_OPTIONS=('FALSE' 'TRUE')
CXX11_ABI_OPTIONS=('FALSE')
CUDA_VERSION='12.8.1' # Fixed as per workflow

TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

# Create a directory for the wheels
mkdir -p dist

# Activate pyenv
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"


for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
    # pyenv requires full version string
    FULL_PYTHON_VERSION=""
    if [[ "$PYTHON_VERSION" == "3.9" ]]; then FULL_PYTHON_VERSION="3.9.19"; fi
    if [[ "$PYTHON_VERSION" == "3.10" ]]; then FULL_PYTHON_VERSION="3.10.14"; fi
    if [[ "$PYTHON_VERSION" == "3.11" ]]; then FULL_PYTHON_VERSION="3.11.9"; fi
    if [[ "$PYTHON_VERSION" == "3.12" ]]; then FULL_PYTHON_VERSION="3.12.3"; fi
    if [[ "$PYTHON_VERSION" == "3.13" ]]; then FULL_PYTHON_VERSION="3.13.0"; fi

    if [ -z "$FULL_PYTHON_VERSION" ]; then
        echo "Unsupported Python version: $PYTHON_VERSION"
        continue
    fi
    pyenv local "$FULL_PYTHON_VERSION"
    echo "Using Python version $(python --version)"

    for TORCH_VERSION in "${TORCH_VERSIONS[@]}"; do
        # Exclude Pytorch < 2.5 with Python 3.13
        if [[ "$TORCH_VERSION" == "2.4.0" && "$PYTHON_VERSION" == "3.13" ]]; then
            echo "Skipping Torch $TORCH_VERSION with Python $PYTHON_VERSION (incompatible)"
            continue
        fi

        export MATRIX_CUDA_VERSION=$(echo "$CUDA_VERSION" | awk -F . '{print $1 $2}')
        export MATRIX_TORCH_VERSION=$(echo "$TORCH_VERSION" | awk -F . '{print $1 "." $2}')
        export WHEEL_CUDA_VERSION=$(echo "$CUDA_VERSION" | awk -F . '{print $1}')
        export MATRIX_PYTHON_VERSION=$(echo "$PYTHON_VERSION" | awk -F . '{print $1 $2}')

        echo "Building with: Python $PYTHON_VERSION, Torch $TORCH_VERSION, CUDA $CUDA_VERSION (env: $MATRIX_CUDA_VERSION), CXX11_ABI $CXX11_ABI"
        echo "MATRIX_TORCH_VERSION: $MATRIX_TORCH_VERSION"
        echo "MATRIX_PYTHON_VERSION: $MATRIX_PYTHON_VERSION"

        # Install PyTorch
        echo "Installing PyTorch..."
        pip install --upgrade pip
        # From workflow: With python 3.13 and torch 2.5.1, unless we update typing-extensions, we get error
        if [[ "$PYTHON_VERSION" == "3.13" && "$TORCH_VERSION" == "2.5.1" ]]; then
            pip install typing-extensions==4.12.2
        else
            # Ensure a compatible version is installed or it will be upgraded
            pip install typing-extensions
        fi
        pip install ninja packaging wheel jinja2

        # Determine TORCH_CUDA_VERSION for PyTorch download URL
        # Logic from publish.yml
        PYTORCH_DOWNLOAD_CUDA_SUFFIX=""
        MINV=""
        MAXV=""
        case $MATRIX_TORCH_VERSION in
            '2.4') MINV=118; MAXV=124 ;;
            '2.5') MINV=118; MAXV=124 ;;
            '2.6') MINV=118; MAXV=126 ;;
            '2.7') MINV=118; MAXV=128 ;;
            *) echo "Unknown MATRIX_TORCH_VERSION: $MATRIX_TORCH_VERSION"; exit 1 ;;
        esac

        if [ "$MATRIX_CUDA_VERSION" -lt 120 ]; then
            PYTORCH_DOWNLOAD_CUDA_SUFFIX="$MINV"
        else
            PYTORCH_DOWNLOAD_CUDA_SUFFIX="$MAXV"
        fi
        echo "PyTorch download CUDA suffix: cu${PYTORCH_DOWNLOAD_CUDA_SUFFIX}"

        if [[ "$TORCH_VERSION" == *"dev"* ]]; then
            echo "Installing pre-release PyTorch version $TORCH_VERSION"
            pip install jinja2 # Dependency for triton?
            # This specific pytorch_triton might need adjustment if TORCH_VERSION changes
            # Example from workflow for 2.6.0.dev20241001
            # pip install https://download.pytorch.org/whl/nightly/pytorch_triton-3.1.0%2Bcf34004b8a-cp${MATRIX_PYTHON_VERSION}-cp${MATRIX_PYTHON_VERSION}-linux_x86_64.whl
            echo "Warning: Dev Pytorch triton wheel URL might be hardcoded or need updates for different dev versions."
            pip install --no-cache-dir --pre "https://download.pytorch.org/whl/nightly/cu${PYTORCH_DOWNLOAD_CUDA_SUFFIX}/torch-${TORCH_VERSION}%2Bcu${PYTORCH_DOWNLOAD_CUDA_SUFFIX}-cp${MATRIX_PYTHON_VERSION}-cp${MATRIX_PYTHON_VERSION}-linux_x86_64.whl"
        else
            echo "Installing release PyTorch version $TORCH_VERSION"
            pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/cu${PYTORCH_DOWNLOAD_CUDA_SUFFIX}"
        fi

        nvcc --version
        python --version
        python -c "import torch; print('PyTorch:', torch.__version__)"
        python -c "import torch; print('CUDA available in PyTorch:', torch.version.cuda)"
        python -c "from torch.utils import cpp_extension; print ('cpp_extension.CUDA_HOME:', cpp_extension.CUDA_HOME)"

        for CXX11_ABI in "${CXX11_ABI_OPTIONS[@]}"; do
            echo "-----------------------------------------------------------------"
            echo "Building wheel for Python $PYTHON_VERSION, Torch $TORCH_VERSION, CXX11_ABI $CXX11_ABI"
            echo "-----------------------------------------------------------------"

            TEMP_DIST_DIR=$(pwd)/dist_temp

            # Clean previous build artifacts
            rm -rf build *.egg-info $TEMP_DIST_DIR

            # Build wheel
            # Setuptools already installed in Dockerfile
            # pip install setuptools==75.8.0
            # pip install ninja packaging wheel # Already installed in Dockerfile
            
            # PATH and LD_LIBRARY_PATH for CUDA should be set by the Dockerfile
            # export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/nvidia/lib64:$PATH
            # export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

            # Limit MAX_JOBS as in the workflow
            # Defaulting to 1 as per 12.8 being >= 12.3 logic in workflow
            MAX_JOBS=128
            # if [ "$MATRIX_CUDA_VERSION" == "128" ]; then # MATRIX_CUDA_VERSION is '128' for CUDA 12.8
            #     MAX_JOBS=1
            # else # This case might not be hit with fixed CUDA_VERSION='12.8.1' but keeping for robustness
            #      MAX_JOBS=2 
            # fi
            echo "Using MAX_JOBS=$MAX_JOBS"

            cd hopper
            MAX_JOBS=$MAX_JOBS FLASH_ATTENTION_FORCE_BUILD="TRUE" FLASH_ATTENTION_FORCE_CXX11_ABI="$CXX11_ABI" python setup.py bdist_wheel --dist-dir=$TEMP_DIST_DIR
            cd ..

            echo "--------------------------------"
            ls -l $TEMP_DIST_DIR
            echo "--------------------------------"

            # Rename wheel
            if [ -d $TEMP_DIST_DIR ] && [ "$(ls -A $TEMP_DIST_DIR)" ]; then
                TMPNAME="cu${WHEEL_CUDA_VERSION}torch${MATRIX_TORCH_VERSION}cxx11abi${CXX11_ABI}"
                ORIGINAL_WHEEL_NAME=$(ls $TEMP_DIST_DIR/*whl | xargs -n 1 basename)
                NEW_WHEEL_NAME=$(echo "$ORIGINAL_WHEEL_NAME" | sed "s/-/+$TMPNAME-/2")

                mv "$TEMP_DIST_DIR/$ORIGINAL_WHEEL_NAME" "dist/$NEW_WHEEL_NAME"
                echo "Built wheel: dist/$NEW_WHEEL_NAME"
                rm -rf $TEMP_DIST_DIR
            else
                echo "Wheel build failed for Python $PYTHON_VERSION, Torch $TORCH_VERSION, CXX11_ABI $CXX11_ABI"
            fi
            echo "Cleaning up build directory..."
            rm -rf build
        done # CXX11_ABI
    done # TORCH_VERSION
done # PYTHON_VERSION

echo "All builds finished. Wheels are in ./dist"
ls -l dist 
