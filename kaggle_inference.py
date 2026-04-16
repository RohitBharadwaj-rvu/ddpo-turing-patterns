import os
import sys

# 1. Ensure TPU Libraries are installed (for Kaggle TPUv4 VM)
def setup_tpu_env():
    if "KAGGLE_URL_BASE" in os.environ:
        try:
            import torch_xla
        except ImportError:
            print(">> Installing PyTorch XLA for TPU support...")
            os.system("pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html")
            print(">> Installed. You may need to restart the session/kernel for XLA to load properly.")

setup_tpu_env()

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig
import matplotlib.pyplot as plt

# 2. Setup Device (TPU vs GPU fallback)
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    is_tpu = True
    print(f">> Successfully connected to TPU: {device}")
except ImportError:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_tpu = False
    print(f">> TPU libraries not found. Falling back to: {device}")

def main():
    print(">> Loading Base Model...")
    # TPUs map bfloat16 extremely well. GPUs use float16.
    dtype_to_use = torch.bfloat16 if is_tpu else torch.float16
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype_to_use,
        safety_checker=None
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    prompts = [
        "a wide shot of a futuristic cyberpunk city, neon lights, 4k",
        "a minimalist living room with large windows, mid-century modern furniture, soft sunlight, highly detailed",
        "a hyper-realistic macro shot of a complex circuit board, glowing traces, intricate details",
        "a field of abstract alien flora with repeating bio-luminescent geometric patterns, trending on artstation"
    ]

    images_per_prompt = 2
    os.makedirs("/kaggle/working/outputs", exist_ok=True)

    print("\n>> PHASE 1: Generating Vanilla (No LoRA) Images...")
    if is_tpu:
        print("   [!] NOTE: The first generation on TPU will take ~3-5 minutes because PyTorch XLA has to compile the Diffusers execution graph. Subsequent generations will be extremely fast.")

    vanilla_results = []
    for i, prompt in enumerate(prompts):
        print(f"   [Vanilla] Prompt {i+1}: '{prompt}'")
        batch_prompts = [prompt] * images_per_prompt
        images = pipeline(batch_prompts, num_inference_steps=30, guidance_scale=6.0).images
        vanilla_results.append(images)

    print("\n>> Configuring LoRA Architecture...")
    lora_config = LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.unet.add_adapter(lora_config)

    print(">> Searching for uploaded LoRA weights...")
    lora_path = None
    for root, _, files in os.walk("/kaggle/input/"):
        for f in files:
            if "unet_lora" in f and f.endswith(".pt"):
                lora_path = os.path.join(root, f)
                break
        if lora_path:
            break

    if not lora_path and os.path.exists("C:\\Users\\rohit\\Downloads\\unet_lora (1).pt"):
         lora_path = "C:\\Users\\rohit\\Downloads\\unet_lora (1).pt"
         
    if lora_path:
        print(f"   Found LoRA at: {lora_path}")
        pipeline.unet.load_state_dict(torch.load(lora_path, map_location="cpu"), strict=False)
        print("   ✅ LoRA weights loaded successfully!")
    else:
        print("   ❌ WARNING: Could not find unet_lora.pt! Phase 2 will just be vanilla again.")

    if is_tpu:
        pipeline = pipeline.to(device)

    print("\n>> PHASE 2: Generating LoRA Images...")
    lora_results = []
    for i, prompt in enumerate(prompts):
        print(f"   [LoRA]    Prompt {i+1}: '{prompt}'")
        batch_prompts = [prompt] * images_per_prompt
        images = pipeline(batch_prompts, num_inference_steps=30, guidance_scale=6.0).images
        lora_results.append(images)

    print("\n>> Plotting Results (Vanilla vs LoRA)...")
    for i, prompt in enumerate(prompts):
        # 4 images total (2 vanilla + 2 lora)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Vanilla
        for j, img in enumerate(vanilla_results[i]):
            out_path = f"/kaggle/working/outputs/prompt_{i+1}_vanilla_{j+1}.png"
            img.save(out_path)
            axes[j].imshow(img)
            axes[j].axis('off')
            axes[j].set_title(f"Vanilla Gen {j+1}")
            
        # LoRA
        for j, img in enumerate(lora_results[i]):
            out_path = f"/kaggle/working/outputs/prompt_{i+1}_lora_{j+1}.png"
            img.save(out_path)
            axes[j+2].imshow(img)
            axes[j+2].axis('off')
            axes[j+2].set_title(f"LoRA Gen {j+1}")
            
        plt.suptitle(f"Prompt: {prompt}")
        plt.tight_layout()
        plt.show() # Will display inline in Kaggle notebook

    print("\n>> All images generated and saved to /kaggle/working/outputs/")

if __name__ == "__main__":
    main()
