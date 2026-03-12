## MAKE SURE TOO INSTALL PYTORCH WITH RESPECT TO YOUR PLATFORM AT https://pytorch.org/get-started/locally/

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
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTokenizer, CLIPTextModel
import re

print("imports finished")


@dataclass
class TrainingConfig:
    image_size = 512 #default 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 10 # how many images to sample during evaluation
    num_epochs = 70 #default 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500#default 500
    save_image_epochs = 10#default 10
    save_model_epochs = 30#default 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "food-test"  # the model name locally and on the HF Hub
    pretrained_model_name_or_path = "openai/clip-vit-base-patch32"  # The text encoder


    push_to_hub = False # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = None
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0



config = TrainingConfig()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.requires_grad_(False)

model = UNet2DConditionModel(
    sample_size=TrainingConfig.image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",

    ),
    cross_attention_dim=768
)




noise_scheduler = DDPMScheduler(num_train_timesteps=1000)


# resize image into same size
preprocess = transforms.Compose(
    [
        transforms.Resize((TrainingConfig.image_size, TrainingConfig.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

#transform data to be ready to train
def transform(instance):
    images = [preprocess(image.convert("RGB")) for image in instance["image"]]
    captions = []


    #for every image in the dataset
    for i in range(len(instance["image"])):

        food_name = instance["food101_class_name"][i]

        raw_text = instance["qwen3_vl_8b_yaml_out"][i]
        if raw_text:
            ingredients = re.sub(r'\s*\(.*?\)', '', raw_text).strip()
        else:
            ingredients = "Various ingrediants"
        food_description = instance["output_label_json"][i]

        prompt = f"Make me {food_name} {food_description} made out of {ingredients}"

        captions.append(prompt)

    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding = "max_length",
        truncation=True,
        return_tensors="pt"

    )
    return {"images": images,
            "input_ids": inputs.input_ids
    }


dataset = load_dataset("mrdbourke/FoodExtract-1k-Vision", split="train")


# Apply the transform to the dataset
dataset.set_transform(transform)

# Initialize trainer
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=TrainingConfig.train_batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def evaluate(config, epoch, model, noise_scheduler, text_encoder, tokenizer, device):
    # Define a prompt to test
    prompt = ["A hamburger with cheese and lettuce", "A delicious slice of pizza with various ingrediants", "Sushi rolls on a plate",
              "Ice cream with chocolate sauce"]
    # Adjust prompt list size to match eval_batch_size
    prompt = prompt[:config.eval_batch_size]

    # 1. Encode Text
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

    # 2. Prepare random noise
    images = torch.randn(
        (len(prompt), model.config.in_channels, model.config.sample_size, model.config.sample_size),
        device=device,
    )

    # 3. Denoising Loop
    model.eval()
    for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
        with torch.no_grad():
            # Apply classifier-free guidance if you wanted to, but for now simple conditioning:
            model_output = model(images, t, encoder_hidden_states=text_embeddings).sample

            # Update images using scheduler
            images = noise_scheduler.step(model_output, t, images).prev_sample

    # 4. Save Images
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()

    # Convert to PIL and save
    image_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(image_dir, exist_ok=True)

    from PIL import Image
    for i, img_array in enumerate(images):
        img = Image.fromarray((img_array * 255).round().astype("uint8"))
        img.save(f"{image_dir}/{epoch:04d}_{prompt[i].replace(' ', '_')}.png")

    model.train()  # Set back to train mode

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

    text_encoder.to(accelerator.device)

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
            input_ids = batch["input_ids"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # We must do this inside torch.no_grad() because we aren't training the text encoder
            with torch.no_grad():
                # The text encoder must be on the same device (GPU) as the input_ids
                # If using accelerator, it handles device placement usually,
                # but ensure text_encoder is moved to accelerator.device before the loop starts
                encoder_hidden_states = text_encoder(input_ids.to(accelerator.device))[0]

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=encoder_hidden_states).sample
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


        unwrapped_model =  accelerator.unwrap_model(model)

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            if (epoch + 1) % TrainingConfig.save_image_epochs == 0 or epoch == TrainingConfig.num_epochs - 1:
                evaluate(TrainingConfig, epoch, unwrapped_model, noise_scheduler, text_encoder, tokenizer, accelerator.device)

            if (epoch + 1) % TrainingConfig.save_model_epochs == 0 or epoch == TrainingConfig.num_epochs - 1:
                unwrapped_model.save_pretrained(os.path.join(config.output_dir, "model"))

                tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
