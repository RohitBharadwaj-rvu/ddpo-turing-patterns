# Kaggle Deployment Guide: DDPO Turing Patterns

This guide provides instructions for deploying the automated `ddpo_turing.py` monolithic script natively in a Kaggle notebook environment with zero manual intervention required once started.

## 1. Setup Your Kaggle Notebook

1. **Create a New Notebook**: Go to Kaggle, click "Create" -> "New Notebook"
2. **Environment Settings**: 
   - Accelerator: `GPU T4 x2` or `GPU P100` (Depending on quota; T4 is sufficient for 16GB FP16)
   - Internet: `Enabled`
   - Persistence: `Files only` (Recommended to preserve your `/working` checkpoint folder)

## 2. Prepare the Notebook Cell

Paste the following bash script into the very first cell of your Kaggle notebook.
This cell will automatically upgrade pip, install dependencies, download the script (if hosted) or write it to disk, and then immediately execute it.

```bash
%%bash
# 1. Update and install required dependencies for TRL & Diffusers DDPO
pip install -U pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
pip install -U accelerate transformers diffusers trl peft bitsandbytes xformers datasets -q

# 2. Ensure weights and artifacts directory exists natively in Kaggle's working dir
mkdir -p /kaggle/working/working

# 3. Create the monolithic python script directly
cat << 'EOF' > /kaggle/working/ddpo_turing.py
import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from trl import DDPOTrainer, DDPOConfig, DefaultDDPOStableDiffusionPipeline

# ... [Copy the full contents of ddpo_turing.py here] ...
# (We assume you upload ddpo_turing.py into Kaggle's /kaggle/input dataset OR simply clone it)
EOF

# If you uploaded 'ddpo_turing.py' automatically as a Kaggle Dataset called 'ddpo-script'
# Copy it over instead of using cat:
# cp /kaggle/input/ddpo-script/ddpo_turing.py /kaggle/working/ddpo_turing.py

# 4. Execute the training non-interactively
# Hugging Face caching defaults might warn if not authenticated, you may want to login if accessing protected models
# python -c "from huggingface_hub import login; login('YOUR_HF_TOKEN')"

echo "Starting deployment..."
cd /kaggle/working/
python ddpo_turing.py
```

## 3. Run and Walk Away

Click **"Run All"** (or execute the cell).

**Execution Flow**:
1. The script automatically fetches dependencies.
2. It launches `ddpo_turing.py`.
3. The internal mathematical sanity checks will ensure that your Turing DoG calculations and PyTorch GPU Hamming Distances are evaluating correctly.
4. DDPO Trainer sets up SD1.5 with FP16 components caching to stay strictly inside the 16GB VRAM bounds.
5. It runs iteratively, tracking the system time constantly. By the end of 27 hours and ~45 minutes (approx. before precisely hitting 28 hours), the loop breaks to safely serialize and save `.safetensors` files to `/kaggle/working/`.

You can just return to Kaggle the next day and download `/kaggle/working/working/ddpo_turing_final` natively!
