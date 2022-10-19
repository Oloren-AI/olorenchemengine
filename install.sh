#!/bin/bash

GPU=0
if  nvidia-smi 2>/dev/null | grep -q "Driver Version"; then
    GPU=1
fi
if [[ $* == *--gpu* ]]; then
    GPU=1
fi

if [[ $(python -c "import sys, os; print(os.path.exists(os.path.join(sys.prefix, 'conda-meta')))") = *True* ]]; then
    echo "Detected conda environment, using conda commands going forward..."
    CONDA=1
else
    echo "Detected non-conda environment, using pip commands going forward..."
    if [[ $OSTYPE == 'darwin'* ]]; then
        which -s brew
        if [[ $? != 0 ]] ; then
            echo "Missing Homebrew on OSX - please install first at https://brew.sh/ and add to path"
            exit 1
        else
            brew install python@3.8
        fi
    fi
fi


# Check if pytorch is installed
if ! python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "1.11.0" ; then
    echo "Pytorch 1.11.0 (required for torch geometric) not found, installing..."

    if [ $GPU == 1 ]; then
        echo "Detected NVIDIA GPU and Drivers, installing CUDA enabled PyTorch with:"
        if [ $CONDA == 1 ]; then
            echo ">> conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch"
            conda install pytorch==1.11.0 cudatoolkit=11.3 -c pytorch
        else
            echo ">> python3.8 -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 install"
            python3.8 -m pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 install
        fi
    else
        echo "Could not detect NVIDIA GPU or drivers, installing with PyTorch without CUDA enabled:"
        if [ $CONDA == 1 ]; then
            echo ">> conda install pytorch==1.11.0 cpuonly -c pytorch"
            conda install pytorch==1.11.0 cpuonly -c pytorch
        else
            echo ">> python3.8 -m pip install torch==1.11.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu install"
            python3.8 -m pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cpu install
        fi
    fi
else
    echo "Pytorch 1.11.0 found, skipping installation..."
fi

# Check if pytorch geometric is installed
if ! python -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader(sys.argv[1]) else 1)" torch_geometric ; then
    echo "Installing torch geometric..."

    if python -c "import torch; print(torch.version.cuda)" 2>/dev/null | grep -q "11.3" ; then
        echo "Found CUDA=11.3 enabled PyTorch, installing torch geometric with GPU support..."
        if [ $CONDA == 1 ]; then
            echo ">> conda install pyg -c pyg"
            conda install -y pyg -c pyg
        else
            echo ">> python3.8 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html"
            python3.8 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
        fi
    else
        if [ $CONDA == 1 ]; then
            echo ">> conda install pyg -c pyg"
            conda install -y pyg -c pyg
        else
            echo "Found PyTorch without CUDA=11.3, installing torch geometric without GPU support..."
            echo ">> python3.8 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html"
            python3.8 -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
        fi
    fi
else
    echo "Pyg installation found, skipping installation..."
fi


if [[ $* == *--docker* ]]; then
        python3.8 -m pip install olorenchemengine
    echo "Docker argument passed, skipping installing chemengine package."
else
    if [[ $* == *--dev* ]]; then
        SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
        python3.8 -m pip install -e $SCRIPT_DIR # install editable copy of the package for dev
    else
        python3.8 -m pip install olorenchemengine
    fi
fi

if [[ $OSTYPE == 'darwin'* ]]; then
  echo "Detected OSX - upgrading pytorch lightning to fix TF issue..."
  python3.8 -m pip install --upgrade pytorch-lightning
fi

echo "Installation succesful - check out https://docs.oloren.ai to get started with OCE!"