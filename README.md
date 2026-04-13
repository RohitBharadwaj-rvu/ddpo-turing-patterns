# DDPO Turing Patterns: One-Click Kaggle Training

This repository provides a self-bootstrapping script for fine-tuning Stable Diffusion with Denoising Diffusion Policy Optimization (DDPO) to generate Turing patterns. It is designed to run non-interactively in Kaggle/Colab with zero setup.

## 🚀 One-Click Deployment (Kaggle/Colab)

The `ddpo_turing.py` script is **monolithic and self-bootstrapping**. You no longer need to run separate shell commands for installations.

1. **Create a New Notebook** in Kaggle or Colab.
2. **Enable GPU Accelerator** (e.g., GPU T4 x2 or P100).
3. **Copy the entire content** of [`ddpo_turing.py`](./ddpo_turing.py).
4. **Paste it into a single cell** and run it.

### What the script does automatically:
- **Environment Check**: Detects if it's running in Kaggle/Colab.
- **Auto-Installation**: Installs `torch`, `diffusers`, `trl`, `peft`, `bitsandbytes`, and `xformers` silently.
- **Sanity Checks**: Performs a mathematical check on the reward function and generates a preflight artifact (`/working/preflight_check.png`).
- **Memory Optimized**: Uses LoRA, 8-bit Adam, and FP16 to fit strictly within 16GB VRAM.
- **Time-Aware**: Constantly monitors execution time. It will automatically stop and save final weights before the 28-hour (or instance-specific) timeout.

## 💾 Output
Final weights and sample grids are saved to `./working/ddpo_turing_final`. In Kaggle, these will be available in the `/kaggle/working` directory for download after the run finishes.
