import os
import sys
import subprocess
import time

# =============================================================================
# 0. Environment Setup (Kaggle/Colab Bootstrap)
# =============================================================================

def setup_environment():
    """Detects environment and installs dependencies if needed."""
    is_kaggle = "KAGGLE_URL_BASE" in os.environ
    is_colab = "COLAB_GPU" in os.environ
    
    if is_kaggle or is_colab:
        print(f">> Detected {'Kaggle' if is_kaggle else 'Colab'} environment.")
        print(">> Checking/Installing dependencies (this may take a minute)...")
        
        pkgs = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "accelerate",
            "transformers",
            "diffusers",
            "trl",
            "peft",
            "bitsandbytes",
            "xformers",
            "datasets"
        ]
        
        for pkg in pkgs:
            try:
                subprocess.run(f"{sys.executable} -m pip install -U {pkg} -q", shell=True, check=True)
            except Exception as e:
                print(f"   Warning: Failed to install {pkg}: {e}")
        
        print(">> Dependencies installed successfully.")
    else:
        print(">> Local/Standard environment detected. Skipping auto-install.")

# Run setup before any heavy imports
setup_environment()

# Now safe to import heavy hitters
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from trl import DDPOTrainer, DDPOConfig, DefaultDDPOStableDiffusionPipeline
from datasets import Dataset

# =============================================================================
# 1. Turing Pattern & Custom Reward Functions
# =============================================================================

def turing_pattern_dog(images, sigma_1=1.0, sigma_2=2.0, threshold=0.0):
    """
    Applies Difference of Gaussians (DoG) as a proxy for Turing patterns.
    images: Tensor of shape (B, C, H, W). Assumes images are properly normalized.
    """
    # Using torchvision's gaussian_blur. It expects images in [0, 1] or similar scale.
    blur_1 = TF.gaussian_blur(images, kernel_size=[11, 11], sigma=[sigma_1, sigma_1])
    blur_2 = TF.gaussian_blur(images, kernel_size=[21, 21], sigma=[sigma_2, sigma_2])
    
    # Difference of Gaussians
    dog = blur_1 - blur_2
    
    # Binarize to clearly separate the patterns
    pattern = (dog > threshold).float()
    return pattern

def pairwise_hamming_distance(patterns):
    """
    Computes pairwise Hamming distance between patterns in a batch.
    patterns: Tensor of shape (B, C, H, W) with values 0 or 1.
    """
    B = patterns.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=patterns.device).expand(B)
    
    # Flatten spatial dimensions
    flat_patterns = patterns.view(B, -1)
    
    rewards = torch.zeros(B, device=patterns.device, dtype=torch.float32)
    for i in range(B):
        # All patterns except i
        others = torch.cat([flat_patterns[:i], flat_patterns[i+1:]], dim=0)
        
        # Absolute difference acts as XOR for {0, 1}
        # Meaning: exactly differing pixels
        diff = torch.abs(flat_patterns[i].unsqueeze(0) - others)
        
        # We want to maximize the difference (high diversity)
        # diff.mean() across all other samples and all pixels gives a percentage (0.0 to 1.0)
        rewards[i] = diff.mean()
        
    return rewards

def custom_reward_fn(images, prompts, metadata):
    """
    TRL DDPOTrainer reward function.
    Given generated images, returns a list of floats as rewards.
    We return higher rewards for high Hamming Distance (diversity) and balanced DoG activation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_tensor = T.ToTensor()
    
    if not isinstance(images, list):
        images = [images]
        
    # Convert PIL Images to tensors
    batch = torch.stack([to_tensor(img) for img in images]).to(device)
    
    # 1. Compute Turing patterns via DoG
    patterns = turing_pattern_dog(batch)
    
    # 2. Compute pairwise Hamming distance
    diversity_rewards = pairwise_hamming_distance(patterns) # between 0.0 and 1.0
    
    # 3. Add balance reward: encourage non-trivial patterns (no plain white/black)
    pattern_means = patterns.view(batch.shape[0], -1).mean(dim=1)
    # peaks at 0.5 (perfectly balanced 0s and 1s)
    balance_rewards = 1.0 - torch.abs(pattern_means - 0.5) * 2.0
    
    total_rewards = diversity_rewards + balance_rewards
    
    return total_rewards.cpu().tolist(), {}

# =============================================================================
# 2. Automated Sanity Checks
# =============================================================================

def run_sanity_checks():
    print("--------------------------------------------------")
    print(">> Running preflight sanity checks...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Detected device: {device}")
    
    try:
        # Dummy batch: 4 random images
        dummy_tensor = torch.rand(4, 3, 512, 512, device=device)
        
        # Test Turing logic
        patterns = turing_pattern_dog(dummy_tensor)
        assert patterns.shape == (4, 3, 512, 512), f"Expected shape (4, 3, 512, 512) got {patterns.shape}"
        assert torch.all(patterns >= 0) and torch.all(patterns <= 1), "Patterns must be binary 0 or 1."
        
        # Test Hamming distance
        distances = pairwise_hamming_distance(patterns)
        assert distances.shape == (4,), f"Expected shape (4,) got {distances.shape}"
        assert torch.all(distances >= 0) and torch.all(distances <= 1.0), "Distances must be bounded [0, 1]."
        
        # Everything passed, export artifact
        os.makedirs("./working", exist_ok=True)
        to_pil = T.ToPILImage()
        to_pil(patterns[0]).save("./working/preflight_check.png")
        print("   Preflight checks passed! Artifact saved to ./working/preflight_check.png")
        print("--------------------------------------------------\n")
    except Exception as e:
        print(f"!!! SANITY CHECK FAILED !!!\n{e}")
        exit(1)

# =============================================================================
# 3. Main DDPO Training execution
# =============================================================================

def prompt_fn():
    """Generates prompt/metadata tuples for DDPO."""
    prompts = [
        "A highly detailed Turing pattern",
        "Reaction-diffusion texture",
        "Intricate organic difference of gaussians"
    ]
    # randomly select
    chosen = prompts[torch.randint(0, len(prompts), (1,)).item()]
    return chosen, {}

def main():
    # 1. First, validate the mathematical pipeline
    run_sanity_checks()

    # 2. Initialize Pipeline & Memory optimizations
    print(">> Initializing DDPO Trainer & Model...")
    
    # We use Runway's SD1.5 base as a lightweight test
    pipeline = DefaultDDPOStableDiffusionPipeline(
        "runwayml/stable-diffusion-v1-5",
        use_lora=True,          # Essential for 16GB
    )
    
    # Prepare DDPOTrainer Config tailored for max VRAM 16GB
    config = DDPOConfig(
        num_epochs=1000,                  # We manage the break programmatically
        train_gradient_accumulation_steps=4,
        sample_num_steps=50,
        sample_batch_size=2,              # Restrict to avoid OOM
        train_batch_size=2,
        sample_num_batches_per_epoch=8,   # Keep epochs short for fine-grained time checking
        per_prompt_stat_tracking=True,
        tracker_project_name="ddpo_turing_patterns",
        mixed_precision="fp16",           # FP16 essential for 16GB limit
        # In TRL 0.8+ gradient checkpointing is commonly set securely inside DDPO or PEFT config
        # We assume pipeline manages the underlying VRAM optimizations.
    )

    trainer = DDPOTrainer(
        config=config,
        reward_function=custom_reward_fn,
        prompt_function=prompt_fn,
        sd_pipeline=pipeline,
    )
    
    # Enable inner memory hacks explicitly if available
    try:
        if hasattr(trainer.sd_pipeline, "enable_xformers_memory_efficient_attention"):
            trainer.sd_pipeline.enable_xformers_memory_efficient_attention()
        if hasattr(trainer.sd_pipeline, "enable_gradient_checkpointing"):
            trainer.sd_pipeline.enable_gradient_checkpointing()
    except Exception as e:
        print("   Proceeding without explicit xformers/checkpointing toggles:", e)

    import json
    
    # 3. Setup Time-Aware and Resume Training Loop
    MAX_RUNTIME_HOURS = 28.0
    SAFETY_MARGIN = 0.25 # Break 15 mins early to save weights securely
    MAX_RUNTIME_SECONDS = (MAX_RUNTIME_HOURS - SAFETY_MARGIN) * 3600
    START_TIME = time.time()
    
    CHECKPOINT_DIR = "./working/checkpoint_latest"
    start_epoch = 0
    
    if os.path.exists(CHECKPOINT_DIR):
        print(f">> Found existing checkpoint at {CHECKPOINT_DIR}. Resuming...")
        try:
            # We explicitly load the accelerator state which includes Optimizer momentum
            trainer.accelerator.load_state(CHECKPOINT_DIR)
            
            # Read metadata to resume epoch counter
            with open(os.path.join(CHECKPOINT_DIR, "state_meta.json"), "r") as f:
                meta = json.load(f)
                start_epoch = meta.get("epoch", 0) + 1
            print(f"   Successfully loaded state. Resuming from Epoch {start_epoch}.")
        except Exception as e:
            print(f"   Failed to load checkpoint state: {e}. Starting fresh.")
    
    print(f">> Starting Time-Aware DDPO Loop. Max execution time: {MAX_RUNTIME_HOURS - SAFETY_MARGIN} hrs.")
    
    try:
        for epoch in range(start_epoch, config.num_epochs):
            elapsed = time.time() - START_TIME
            
            if elapsed >= MAX_RUNTIME_SECONDS:
                print(f"\n>> Time limit reached: {elapsed/3600:.2f} hrs. Safely breaking loop to save final weights.")
                break
                
            print(f"--- Epoch {epoch} | Time Elapsed: {elapsed/3600:.2f} hrs ---")
            
            # Step executes the environment rollout and policy unrolling
            trainer.step(epoch, epoch)
            
            # Checkpoint roughly every 5 epochs
            if (epoch + 1) % 5 == 0:
                ckpt_path = f"./working/ddpo_turing_epoch_{epoch}"
                trainer.save_pretrained(ckpt_path)
                print(f"   Saved checkpoint -> {ckpt_path}")

    except KeyboardInterrupt:
        print("\n>> Stop button pressed! Safely Pausing & Saving State...")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        # 1. Save standard model LoRA adapter
        trainer.save_pretrained(CHECKPOINT_DIR)
        # 2. Save accelerator state (Optimizers, Schedulers, RNG)
        trainer.accelerator.save_state(CHECKPOINT_DIR)
        # 3. Save our manual loop epoch metadata
        with open(os.path.join(CHECKPOINT_DIR, "state_meta.json"), "w") as f:
            json.dump({"epoch": epoch}, f)
        print(f">> State seamlessly saved at Epoch {epoch}. You can re-run this cell to resume.")
        return # Exit the main function early, entirely skipping final push since we are "paused"

    finally:
        # If we broke naturally (not interrupted), push final weights
        # Note: If KeyboardInterrupt triggered, we bypass this block because of `return`. Wait! `finally` ALWAYS executes even with `return`.
        pass
        
    # Final weights push (Only executes if we naturally finished or hit time limit)
    final_path = f"./working/ddpo_turing_final_{int(time.time())}"
    trainer.save_pretrained(final_path)
    print(f">> Finished execution. Full pipeline LORA saved to {final_path}")

if __name__ == "__main__":
    main()
