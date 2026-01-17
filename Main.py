from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from accelerate import notebook_launcher
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
print("imports finnished")

@dataclass
class TrainingConfig:
    image_size = 64 #default 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 1 #default 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 0#default 500
    save_image_epochs = 1#default 10
    save_model_epochs = 1#default 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "food-test"  # the model name locally and on the HF Hub

    push_to_hub = False # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

# 1. Initialize Config FIRST
config = TrainingConfig()

# 2. Dataset Selection
config.dataset_name = "mrdbourke/FoodExtract-1k-Vision"
dataset = load_dataset(config.dataset_name, split="train")


# resize image into same size
preprocess = transforms.Compose(
    [
        transforms.Resize((TrainingConfig.image_size, TrainingConfig.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

# Apply the transform to the dataset
dataset.set_transform(transform)



# Initialize trainer
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=TrainingConfig.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=TrainingConfig.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

import torch
from PIL import Image
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, pipeline):
    # Generate a few images to check progress
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Save images locally
    image_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(image_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"{image_dir}/{epoch:04d}_{i}.png")


def train_loop(TrainingConfig, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=TrainingConfig.mixed_precision,
        gradient_accumulation_steps=TrainingConfig.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(TrainingConfig.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if TrainingConfig.output_dir is not None:
            os.makedirs(TrainingConfig.output_dir, exist_ok=True)
        if TrainingConfig.push_to_hub:
            repo_id = create_repo(
                repo_id=TrainingConfig.hub_model_id or Path(TrainingConfig.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(TrainingConfig.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % TrainingConfig.save_image_epochs == 0 or epoch == TrainingConfig.num_epochs - 1:
                evaluate(TrainingConfig, epoch, pipeline)

            if (epoch + 1) % TrainingConfig.save_model_epochs == 0 or epoch == TrainingConfig.num_epochs - 1:
                if TrainingConfig.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=TrainingConfig.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(TrainingConfig.output_dir)


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
