import os
import sys
import subprocess
import time
import math

# =============================================================================
# 0. Environment Setup (Kaggle/Colab Bootstrap) — NO trl!
# =============================================================================

def setup_environment():
    """Detects environment and installs dependencies if needed."""
    is_kaggle = "KAGGLE_URL_BASE" in os.environ
    is_colab = "COLAB_GPU" in os.environ
    
    if is_kaggle or is_colab:
        print(f">> Detected {'Kaggle' if is_kaggle else 'Colab'} environment.")
        print(">> Checking/Installing dependencies (this may take a minute)...")
        
        pkgs = [
            "accelerate",
            "transformers",
            "diffusers",
            "peft",
            "bitsandbytes",
            "datasets",
            "matplotlib"
        ]
        
        # Do NOT use -U (upgrade). Kaggle's pre-installed PyTorch is compiled
        # for P100 (sm_60). Upgrading any dep can pull a newer PyTorch that
        # drops sm_60 support and causes "no kernel image" CUDA errors.
        for pkg in pkgs:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)
            except Exception as e:
                print(f"   Warning: Failed to install {pkg}: {e}")
        
        print(">> Dependencies installed successfully.")
    else:
        print(">> Local/Standard environment detected. Skipping auto-install.")

setup_environment()

# Heavy imports
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import json
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig

# =============================================================================
# 1. DDPO Core: DDIM Step with Log Probability
#    Reference: github.com/kvablack/ddpo-pytorch
# =============================================================================

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def ddim_step_with_logprob(scheduler, model_output, timestep, sample, eta=1.0, prev_sample=None):
    """
    DDIM reverse step returning (next_sample, log_prob).
    The log_prob measures how likely the transition is under the current policy.
    This is the mathematical heart of DDPO.
    """
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=sample.device)
    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)

    prev_t = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_t = torch.clamp(prev_t, 0, scheduler.config.num_train_timesteps - 1)

    alpha_t = _left_broadcast(scheduler.alphas_cumprod.gather(0, timestep.cpu()), sample.shape).to(sample.device)
    alpha_prev = torch.where(
        prev_t.cpu() >= 0,
        scheduler.alphas_cumprod.gather(0, prev_t.cpu()),
        scheduler.final_alpha_cumprod,
    )
    alpha_prev = _left_broadcast(alpha_prev, sample.shape).to(sample.device)
    beta_t = 1 - alpha_t

    # Predict x_0
    if scheduler.config.prediction_type == "v_prediction":
        pred_x0 = alpha_t ** 0.5 * sample - beta_t ** 0.5 * model_output
        pred_eps = alpha_t ** 0.5 * model_output + beta_t ** 0.5 * sample
    else:  # epsilon (default)
        pred_x0 = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
        pred_eps = model_output

    if scheduler.config.clip_sample:
        pred_x0 = pred_x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    # Variance & std
    beta_prev = 1 - alpha_prev
    variance = (beta_prev / beta_t) * (1 - alpha_t / alpha_prev)
    std = eta * variance ** 0.5
    std = _left_broadcast(std, sample.shape).to(sample.device) if std.ndim < sample.ndim else std

    # Direction + mean
    direction = (1 - alpha_prev - std ** 2) ** 0.5 * pred_eps
    mean = alpha_prev ** 0.5 * pred_x0 + direction

    if prev_sample is None:
        prev_sample = mean + std * torch.randn_like(model_output)

    # Log prob of prev_sample under Gaussian(mean, std)
    log_prob = (
        -((prev_sample.detach() - mean) ** 2) / (2 * std ** 2)
        - torch.log(std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=sample.device)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.to(sample.dtype), log_prob


# =============================================================================
# 2. DDPO Core: Sampling with Log Probability Tracking
# =============================================================================

@torch.no_grad()
def sample_with_logprob(pipeline, prompt_embeds, neg_prompt_embeds, num_steps, guidance_scale, eta, device):
    """Full denoising pass that records latents & log_probs at every step."""
    scheduler = pipeline.scheduler
    unet = pipeline.unet
    B = prompt_embeds.shape[0]

    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    latents = torch.randn(B, 4, 64, 64, device=device, dtype=prompt_embeds.dtype)
    latents = latents * scheduler.init_noise_sigma

    all_latents = [latents]
    all_log_probs = []
    embeds = torch.cat([neg_prompt_embeds, prompt_embeds])

    for t in timesteps:
        lat_in = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        noise_pred = unet(lat_in, t, encoder_hidden_states=embeds).sample
        uncond, cond = noise_pred.chunk(2)
        noise_pred = uncond + guidance_scale * (cond - uncond)

        latents, lp = ddim_step_with_logprob(scheduler, noise_pred, t, latents, eta=eta)
        all_latents.append(latents)
        all_log_probs.append(lp)

    # Decode
    imgs_t = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    imgs_t = (imgs_t / 2 + 0.5).clamp(0, 1)
    pil_imgs = []
    for img in imgs_t:
        arr = (img.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pil_imgs.append(Image.fromarray(arr))

    return pil_imgs, all_latents, all_log_probs


# =============================================================================
# 3. Turing Pattern & Custom Reward Functions
#    - Blur-Sharpen cycle (reaction-diffusion) for pattern extraction
#    - Pairwise Hamming distance rewards DIVERSITY across same-prompt generations
# =============================================================================

def _rgb_to_lab_L(images):
    """Converts RGB tensor (B,3,H,W) in [0,1] to LAB L-channel (B,1,H,W) in [0,1]."""
    # Step 1: sRGB gamma → linear RGB
    linear = torch.where(images <= 0.04045, images / 12.92,
                         ((images + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]

    # Step 2: Linear RGB → CIE XYZ (Y = luminance)
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b  # (B, 1, H, W)

    # Step 3: XYZ Y → LAB L*
    # L* = 116 * f(Y/Yn) - 16, where Yn=1.0 (D65)
    fy = torch.where(Y > 0.008856, Y ** (1.0 / 3.0), 7.787 * Y + 16.0 / 116.0)
    L = 116.0 * fy - 16.0  # Range [0, 100]

    return L / 100.0  # Normalize to [0, 1]

def turing_pattern_blur_sharpen(images, iterations=5, blur_radius=2,
                                sharpen_strength=1.0, threshold=0.5):
    """
    Extracts Turing-like patterns via iterative blur-sharpen cycles.
    Uses LAB L-channel (perceptual luminosity), blur radius 2, medium sharpen.
    """
    # Extract perceptual luminosity via LAB L-channel
    x = _rgb_to_lab_L(images)  # (B, 1, H, W) in [0, 1]

    kernel_size = blur_radius * 2 + 1  # radius 2 → kernel 5x5
    sigma = blur_radius * 0.5 + 0.5    # sigma matched to radius

    for _ in range(iterations):
        blurred = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma])
        x = x + sharpen_strength * (x - blurred)   # unsharp mask (reaction)
        x = x.clamp(0, 1)

    # Binarize to extract clean pattern structure
    pattern = (x > threshold).float()
    return pattern

def pairwise_hamming_distance(patterns):
    """Rewards images whose patterns are MOST DIFFERENT from each other."""
    B = patterns.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=patterns.device).expand(B)
    flat = patterns.view(B, -1)
    rewards = torch.zeros(B, device=patterns.device)
    for i in range(B):
        others = torch.cat([flat[:i], flat[i+1:]], dim=0)
        rewards[i] = torch.abs(flat[i].unsqueeze(0) - others).mean()
    return rewards

def custom_reward_fn(images, prompts, metadata):
    """All images share the SAME prompt. Reward = diversity + balance."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    to_tensor = T.ToTensor()
    if not isinstance(images, list):
        images = [images]
    batch = torch.stack([to_tensor(img) for img in images]).to(device)
    patterns = turing_pattern_blur_sharpen(batch)
    diversity = pairwise_hamming_distance(patterns)
    means = patterns.view(batch.shape[0], -1).mean(dim=1)
    balance = 1.0 - torch.abs(means - 0.5) * 2.0
    return (diversity + balance).cpu().tolist(), {}


# =============================================================================
# 4. Automated Sanity Checks (UNCHANGED)
# =============================================================================

def run_sanity_checks():
    print("--------------------------------------------------")
    print(">> Running preflight sanity checks...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Detected device: {device}")
    try:
        dummy = torch.rand(4, 3, 512, 512, device=device)
        pat = turing_pattern_blur_sharpen(dummy)
        assert pat.shape == (4, 1, 512, 512), f"Expected (4,1,512,512) got {pat.shape}"
        dist = pairwise_hamming_distance(pat)
        assert dist.shape == (4,) and torch.all(dist >= 0) and torch.all(dist <= 1.0)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        T.ToPILImage()(pat[0]).save(os.path.join(OUTPUT_DIR, "preflight_check.png"))
        print("   Preflight checks passed!")
        print("--------------------------------------------------\n")
    except Exception as e:
        print(f"!!! SANITY CHECK FAILED !!!\n{e}")
        exit(1)


# =============================================================================
# 5. Prompt Dataset (UNCHANGED)
# =============================================================================

prompt_iterator = None

def get_prompt_iterator():
    from datasets import load_dataset
    print(">> Loading Gustavosta/Stable-Diffusion-Prompts dataset...")
    ds = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="train").shuffle(seed=42)
    def _iter():
        for item in ds:
            yield item["Prompt"]
    return _iter()

def prompt_fn():
    global prompt_iterator
    if prompt_iterator is None:
        prompt_iterator = get_prompt_iterator()
    try:
        return next(prompt_iterator), {}
    except StopIteration:
        prompt_iterator = get_prompt_iterator()
        return next(prompt_iterator), {}


# Output directory: /kaggle/working/ persists after committed runs
OUTPUT_DIR = "/kaggle/working" if "KAGGLE_URL_BASE" in os.environ else "./working"

# =============================================================================
# 6. Main DDPO Training Loop — From Scratch, No trl
# =============================================================================

def _save_state(pipeline, optimizer, epoch, history, checkpoint_dir):
    """Saves LoRA weights, optimizer, and metadata to checkpoint_dir."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    lora_state = {k: v.cpu() for k, v in pipeline.unet.state_dict().items() if "lora" in k.lower()}
    torch.save(lora_state, os.path.join(checkpoint_dir, "unet_lora.pt"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    with open(os.path.join(checkpoint_dir, "state_meta.json"), "w") as f:
        json.dump({"epoch": epoch, "history": history}, f)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_sanity_checks()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model Setup ──────────────────────────────────────────────────────
    print(">> Loading Stable Diffusion v1.5 (this may take a few minutes)...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)

    # Freeze everything except UNet LoRA
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    # Inject LoRA into UNet attention layers
    lora_config = LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.unet.add_adapter(lora_config)
    pipeline.unet.enable_gradient_checkpointing()

    trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    print(f"   Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    # Optimizer
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=3e-4)
        print("   Using 8-bit AdamW optimizer.")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)
        print("   Using standard AdamW optimizer.")

    # ── Hyperparameters ──────────────────────────────────────────────────
    NUM_EPOCHS         = 1000
    SAMPLE_BATCH_SIZE  = 4     # Generate 4 images of the SAME prompt per batch
    BATCHES_PER_EPOCH  = 4     # 4 different prompts per epoch, 4 images each = 16 total
    NUM_STEPS          = 20    # DDIM denoising steps
    GUIDANCE_SCALE     = 5.0
    ETA                = 1.0   # Must be >0 for valid log probs
    CLIP_RANGE         = 2e-3  # PPO clip range (wider than 1e-4 for stronger signal)
    ADV_CLIP           = 5.0

    # ── Time-Aware + Resume ──────────────────────────────────────────────
    # Kaggle committed GPU runs have a 9hr limit. Stop at 8.5hrs to save.
    MAX_RUNTIME_SECONDS = (8.5 - 0.25) * 3600  # 8.25 hrs effective
    START_TIME = time.time()

    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoint_latest")
    start_epoch = 0
    history = {"loss": [], "reward": []}

    if os.path.exists(CHECKPOINT_DIR):
        print(f">> Found checkpoint at {CHECKPOINT_DIR}. Resuming...")
        try:
            lora_path = os.path.join(CHECKPOINT_DIR, "unet_lora.pt")
            if os.path.exists(lora_path):
                pipeline.unet.load_state_dict(torch.load(lora_path, map_location=device), strict=False)
            opt_path = os.path.join(CHECKPOINT_DIR, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            with open(os.path.join(CHECKPOINT_DIR, "state_meta.json"), "r") as f:
                meta = json.load(f)
                start_epoch = meta.get("epoch", 0) + 1
                history = meta.get("history", {"loss": [], "reward": []})
            print(f"   Resumed from Epoch {start_epoch}.")
        except Exception as e:
            print(f"   Failed to load checkpoint: {e}. Starting fresh.")

    # Pre-compute negative prompt embeddings (empty string)
    neg_ids = pipeline.tokenizer(
        [""] * SAMPLE_BATCH_SIZE, return_tensors="pt",
        padding="max_length", truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    neg_embeds = pipeline.text_encoder(neg_ids)[0]

    print(f">> Starting DDPO Loop. Max runtime: 8.25 hrs (Kaggle committed).")

    # ── Training Loop ────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            elapsed = time.time() - START_TIME
            if elapsed >= MAX_RUNTIME_SECONDS:
                print(f"\n>> Time limit reached ({elapsed/3600:.2f} hrs). Breaking.")
                break

            print(f"--- Epoch {epoch} | Elapsed: {elapsed/3600:.2f} hrs ---")

            # ══════════ PHASE 1: SAMPLING (no gradients) ══════════
            pipeline.unet.eval()
            all_samples = []
            all_rewards_list = []

            for batch_idx in range(BATCHES_PER_EPOCH):
                # ALL images in this batch share the SAME prompt
                single_prompt = prompt_fn()[0]
                prompts = [single_prompt] * SAMPLE_BATCH_SIZE

                prompt_ids = pipeline.tokenizer(
                    prompts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                images, latents_list, log_probs_list = sample_with_logprob(
                    pipeline, prompt_embeds, neg_embeds, NUM_STEPS, GUIDANCE_SCALE, ETA, device,
                )

                rewards_val, _ = custom_reward_fn(images, prompts, {})
                rewards_t = torch.tensor(rewards_val, device=device, dtype=torch.float32)
                all_rewards_list.append(rewards_t)

                all_samples.append({
                    "prompt_embeds": prompt_embeds,
                    "neg_embeds": neg_embeds,
                    "latents": torch.stack(latents_list[:-1], dim=1),      # (B, T, 4, 64, 64)
                    "next_latents": torch.stack(latents_list[1:], dim=1),  # (B, T, 4, 64, 64)
                    "log_probs": torch.stack(log_probs_list, dim=1),       # (B, T)
                    "timesteps": pipeline.scheduler.timesteps,             # (T,)
                    "rewards": rewards_t,
                })

            # Compute advantages (normalized rewards)
            all_rewards = torch.cat(all_rewards_list)
            advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)
            idx = 0
            for s in all_samples:
                bs = s["rewards"].shape[0]
                s["advantages"] = advantages[idx:idx+bs]
                idx += bs

            # ══════════ PHASE 2: TRAINING (with gradients) ══════════
            pipeline.unet.train()
            epoch_loss = 0.0
            n_updates = 0

            for sample in all_samples:
                optimizer.zero_grad()
                embeds = torch.cat([sample["neg_embeds"], sample["prompt_embeds"]])
                num_t = sample["timesteps"].shape[0]
                adv = torch.clamp(sample["advantages"], -ADV_CLIP, ADV_CLIP)

                for t_idx in range(num_t):
                    lat = sample["latents"][:, t_idx]
                    next_lat = sample["next_latents"][:, t_idx]
                    t = sample["timesteps"][t_idx]
                    old_lp = sample["log_probs"][:, t_idx]

                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        lat_in = torch.cat([lat] * 2)
                        lat_in = pipeline.scheduler.scale_model_input(lat_in, t)
                        noise_pred = pipeline.unet(lat_in, t, encoder_hidden_states=embeds).sample
                        uncond, cond = noise_pred.chunk(2)
                        noise_pred = uncond + GUIDANCE_SCALE * (cond - uncond)

                    _, new_lp = ddim_step_with_logprob(
                        pipeline.scheduler, noise_pred, t, lat, eta=ETA, prev_sample=next_lat,
                    )

                    ratio = torch.exp(new_lp - old_lp)
                    loss = torch.mean(torch.maximum(
                        -adv * ratio,
                        -adv * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE),
                    ))

                    loss.backward()
                    epoch_loss += loss.item()
                    n_updates += 1

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            # ══════════ PHASE 3: TELEMETRY ══════════
            avg_loss = epoch_loss / max(n_updates, 1)
            avg_reward = all_rewards.mean().item()
            grad_norm = sum(p.grad.norm().item() ** 2 for p in trainable_params if p.grad is not None) ** 0.5
            history["loss"].append(avg_loss)
            history["reward"].append(avg_reward)
            print(f"   Loss: {avg_loss:.8f} | Reward: {avg_reward:.4f} | Grad Norm: {grad_norm:.4f}")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history["loss"], color="red"); ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.grid(True)
                ax2.plot(history["reward"], color="green"); ax2.set_title("Turing Reward"); ax2.set_xlabel("Epoch"); ax2.grid(True)
                plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "training_progress.png")); plt.close()
            except Exception as e:
                print(f"   [Warning] Plot failed: {e}")

            # ══════════ PHASE 4: PERIODIC CHECKPOINT ══════════
            if (epoch + 1) % 10 == 0:
                _save_state(pipeline, optimizer, epoch, history, CHECKPOINT_DIR)
                print(f"   Checkpoint saved at Epoch {epoch}")

    except KeyboardInterrupt:
        print("\n>> Stop button pressed!")

    finally:
        # ALWAYS save state — whether time-limit, interrupt, or error.
        # This guarantees outputs persist in /kaggle/working/ after committed runs.
        print(">> Saving final state...")
        _save_state(pipeline, optimizer, epoch, history, CHECKPOINT_DIR)

        # Also save a clean copy of just the LoRA weights
        final_dir = os.path.join(OUTPUT_DIR, "ddpo_final")
        os.makedirs(final_dir, exist_ok=True)
        lora_state = {k: v.cpu() for k, v in pipeline.unet.state_dict().items() if "lora" in k.lower()}
        torch.save(lora_state, os.path.join(final_dir, "unet_lora.pt"))

        # Save training history as standalone JSON
        with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
            json.dump({"epoch": epoch, "history": history}, f)

        print(f">> All outputs saved to {OUTPUT_DIR}:")
        print(f"   - checkpoint_latest/  (resume state)")
        print(f"   - ddpo_final/         (final LoRA weights)")
        print(f"   - training_progress.png")
        print(f"   - training_history.json")
        print(f"   Total epochs completed: {epoch + 1}")

if __name__ == "__main__":
    main()
