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
# =============================================================================

def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def ddim_step_with_logprob(scheduler, model_output, timestep, sample, eta=1.0, prev_sample=None):
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

    if scheduler.config.prediction_type == "v_prediction":
        pred_x0 = alpha_t ** 0.5 * sample - beta_t ** 0.5 * model_output
        pred_eps = alpha_t ** 0.5 * model_output + beta_t ** 0.5 * sample
    else:
        pred_x0 = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
        pred_eps = model_output

    if scheduler.config.clip_sample:
        pred_x0 = pred_x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    beta_prev = 1 - alpha_prev
    variance = (beta_prev / beta_t) * (1 - alpha_t / alpha_prev)
    std = eta * variance ** 0.5
    std = _left_broadcast(std, sample.shape).to(sample.device) if std.ndim < sample.ndim else std

    direction = (1 - alpha_prev - std ** 2) ** 0.5 * pred_eps
    mean = alpha_prev ** 0.5 * pred_x0 + direction

    if prev_sample is None:
        prev_sample = mean + std * torch.randn_like(model_output)

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

    imgs_t = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    imgs_t = (imgs_t / 2 + 0.5).clamp(0, 1)
    pil_imgs = []
    for img in imgs_t:
        arr = (img.permute(1, 2, 0).cpu().float().numpy() * 255).astype(np.uint8)
        pil_imgs.append(Image.fromarray(arr))

    return pil_imgs, all_latents, all_log_probs


# =============================================================================
# 3. Turing Pattern & Custom Reward Functions
# =============================================================================

def _rgb_to_lab_L(images):
    """Converts RGB tensor (B,3,H,W) in [0,1] to LAB L-channel (B,1,H,W) in [0,1]."""
    linear = torch.where(images <= 0.04045, images / 12.92,
                         ((images + 0.055) / 1.055) ** 2.4)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fy = torch.where(Y > 0.008856, Y ** (1.0 / 3.0), 7.787 * Y + 16.0 / 116.0)
    L = 116.0 * fy - 16.0
    return L / 100.0

def turing_pattern_blur_sharpen(images, iterations=150, blur_radius=2,
                                sharpen_strength=1.0, threshold=0.5):
    x = _rgb_to_lab_L(images)
    kernel_size = blur_radius * 2 + 1
    sigma = blur_radius * 0.5 + 0.5

    for _ in range(iterations):
        blurred = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma])
        x = x + sharpen_strength * (x - blurred)
        x = x.clamp(0, 1)

    pattern = (x > threshold).float()
    return pattern

def pairwise_hamming_distance(patterns):
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
# 4. Sanity Checks
# =============================================================================

OUTPUT_DIR = "/kaggle/working" if "KAGGLE_URL_BASE" in os.environ else "./working"

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
# 5. Prompt Dataset
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


# =============================================================================
# 6. TEST: Single Epoch DDPO — quick validation run
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_sanity_checks()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">> Loading Stable Diffusion v1.5...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.unet.add_adapter(lora_config)
    pipeline.unet.enable_gradient_checkpointing()

    trainable_params = [p for p in pipeline.unet.parameters() if p.requires_grad]
    print(f"   Trainable LoRA parameters: {sum(p.numel() for p in trainable_params):,}")

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=3e-4)
    except ImportError:
        optimizer = torch.optim.AdamW(trainable_params, lr=3e-4)

    # ── Test Hyperparameters (1 epoch, 1 batch) ──────────────────────────
    SAMPLE_BATCH_SIZE  = 4
    BATCHES_PER_EPOCH  = 1     # Just 1 batch for quick test
    NUM_STEPS          = 20
    GUIDANCE_SCALE     = 5.0
    ETA                = 1.0
    CLIP_RANGE         = 2e-3
    ADV_CLIP           = 5.0

    neg_ids = pipeline.tokenizer(
        [""] * SAMPLE_BATCH_SIZE, return_tensors="pt",
        padding="max_length", truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    neg_embeds = pipeline.text_encoder(neg_ids)[0]

    print(">> Running 1 test epoch...")
    t0 = time.time()

    # ══════════ SAMPLING ══════════
    pipeline.unet.eval()
    single_prompt = prompt_fn()[0]
    prompts = [single_prompt] * SAMPLE_BATCH_SIZE
    print(f"   Prompt: \"{single_prompt[:80]}...\"")

    prompt_ids = pipeline.tokenizer(
        prompts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=pipeline.tokenizer.model_max_length,
    ).input_ids.to(device)
    prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

    images, latents_list, log_probs_list = sample_with_logprob(
        pipeline, prompt_embeds, neg_embeds, NUM_STEPS, GUIDANCE_SCALE, ETA, device,
    )

    # Save generated images
    for i, img in enumerate(images):
        img.save(os.path.join(OUTPUT_DIR, f"test_gen_{i}.png"))
    print(f"   Saved {len(images)} generated images to {OUTPUT_DIR}")

    rewards_val, _ = custom_reward_fn(images, prompts, {})
    rewards_t = torch.tensor(rewards_val, device=device, dtype=torch.float32)
    print(f"   Rewards: {[f'{r:.4f}' for r in rewards_val]}")
    print(f"   Mean reward: {rewards_t.mean().item():.4f}")

    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    sample = {
        "prompt_embeds": prompt_embeds,
        "neg_embeds": neg_embeds,
        "latents": torch.stack(latents_list[:-1], dim=1),
        "next_latents": torch.stack(latents_list[1:], dim=1),
        "log_probs": torch.stack(log_probs_list, dim=1),
        "timesteps": pipeline.scheduler.timesteps,
        "advantages": advantages,
    }

    # ══════════ TRAINING ══════════
    pipeline.unet.train()
    optimizer.zero_grad()
    embeds = torch.cat([neg_embeds, prompt_embeds])
    adv = torch.clamp(advantages, -ADV_CLIP, ADV_CLIP)
    epoch_loss = 0.0

    for t_idx in range(sample["timesteps"].shape[0]):
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

    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in trainable_params if p.grad is not None) ** 0.5
    elapsed = time.time() - t0

    print(f"\n>> TEST EPOCH COMPLETE in {elapsed:.1f}s")
    print(f"   Loss:      {epoch_loss / NUM_STEPS:.8f}")
    print(f"   Grad Norm: {grad_norm:.4f}")
    print(f"   Reward:    {rewards_t.mean().item():.4f}")
    print(f"   Images saved to {OUTPUT_DIR}/test_gen_*.png")
    print(">> Everything works! Ready for full training run.")

if __name__ == "__main__":
    main()
