# Mamba Text Classification

A Mamba-based text classification implementation using the original [state-spaces/mamba](https://github.com/state-spaces/mamba) library.

The notebook trains a Mamba classifier on the AG News dataset and includes training, evaluation, and visualization of Mamba's internal operations.

## Requirements

- Linux (Ubuntu recommended)
- Python 3.10
- NVIDIA GPU with CUDA support
- CUDA 12.4
- JupyterLab or Jupyter Notebook

### Tested Environment

The following configuration was tested successfully:

| Package       | Version     |
| ------------- | ----------- |
| Python        | 3.10.14     |
| PyTorch       | 2.6.0+cu124 |
| CUDA          | 12.4        |
| Mamba SSM     | 2.2.4       |
| causal-conv1d | 1.4.0       |
| Transformers  | 4.44.2      |
| Datasets      | 3.6.0       |
| NumPy         | 2.2.6       |
| Triton        | 3.2.0       |
| einops        | 0.8.2       |

Other dependencies include `ninja`, `packaging`, `matplotlib`, and `ipykernel`.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

### 2. Create a Virtual Environment

Use Python 3.10 to create an isolated environment:

```bash
python3.10 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify the Python version:

```bash
python --version
```

Expected Python version: `3.10.x`

### 3. Install PyTorch

Install PyTorch 2.6.0 with CUDA 12.4 support:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Required Dependencies

```bash
python -m pip install \
    numpy==2.2.6 \
    transformers==4.44.2 \
    datasets==3.6.0 \
    causal-conv1d==1.4.0 \
    einops==0.8.2 \
    ninja packaging wheel \
    matplotlib ipykernel
```

### 5. Install Mamba SSM

Install the prebuilt Mamba 2.2.4 wheel compatible with Python 3.10 and PyTorch 2.6:

```bash
python -m pip install --no-deps \
"https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

This wheel was used successfully in the tested environment.

**Note:** The wheel is specific to Linux x86_64, Python 3.10, and the corresponding PyTorch/CUDA configuration.

### 6. Verify Installation

```bash
python -c "import torch, mamba_ssm; print('PyTorch:', torch.__version__); print('Mamba:', mamba_ssm.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.is_available())"
```

Expected output:

```text
PyTorch: 2.6.0+cu124
Mamba: 2.2.4
CUDA: 12.4
GPU: True
```

## Configure Jupyter Kernel

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
    --name=mamba-kernel \
    --display-name="Mamba (Python 3.10)"
```

Open JupyterLab:

```bash
jupyter lab
```

Open `Mamba_Midterm.ipynb` and select **Mamba (Python 3.10)** as the kernel.

## GPU Configuration

The notebook uses CUDA for model execution.

If multiple GPUs are available, specify which GPU to use before importing PyTorch or initializing CUDA.

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

Change `"0"` to the physical GPU index you want to use.

When only one GPU is visible, PyTorch identifies it as `cuda:0`.

**Important:** Restart the notebook kernel after changing `CUDA_VISIBLE_DEVICES`.

## Run the Notebook

1. Open `Mamba_Midterm.ipynb`.
2. Select the configured Jupyter kernel.
3. Confirm the correct GPU is selected.
4. Run the notebook cells in order.

The notebook performs:

- AG News dataset loading
- BERT tokenization
- Data preparation and batching
- Mamba classifier initialization
- Model training
- Validation and test evaluation
- Training loss and accuracy plots
- Mamba internal state analysis
- Selective scan verification
- Delta visualization

The AG News dataset and BERT tokenizer are loaded through Hugging Face. Internet access or an existing local cache may be required.

## Model Configuration

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Dataset                 | AG News           |
| Tokenizer               | bert-base-uncased |
| Mamba Layers            | 2                 |
| Embedding Dimension     | 128               |
| State Dimension         | 16                |
| Convolution Width       | 4                 |
| Maximum Sequence Length | 128               |
| Batch Size              | 64                |
| Epochs                  | 5                 |
| Optimizer               | Adam              |
| Learning Rate           | 0.001             |
| Loss Function           | CrossEntropyLoss  |

## Notes

- The model is trained from scratch.
- The BERT tokenizer is used only for tokenization.
- GPU support is required for the original Mamba CUDA implementation.
- The notebook contains saved outputs and visualizations.
- Model checkpoints (`.pt`, `.pth`) are excluded from the repository.
- Running the full notebook retrains the model and overwrites `mamba_trained.pt`.
- The provided Mamba wheel is not compatible with native macOS. Use a supported Linux CUDA environment.
- Package compatibility may differ across Python, PyTorch, and CUDA versions.

## References

- [Mamba GitHub Repository](https://github.com/state-spaces/mamba)
- [AG News Dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
