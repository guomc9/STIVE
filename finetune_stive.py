import torch
from torch.utils.data import DataLoader
from t2v.models.unet_3d_condition import UNet3DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, DDIMScheduler
from peft import LoraConfig, get_peft_model


def get_cached_set(loader: DataLoader, vae: AutoencoderKL):    
    cached_latents = []
    cached_prompts = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Caching Latents"):
            pixel_values = batch['pixel_values'].to(vae.device, dtype=torch.float16)
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            cached_latents.append(latents.detach().cpu())
            cached_prompts.extend(batch['prompts'])
            del pixel_values, batch, latents
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    cached_latents = torch.cat(cached_latents, dim=0)
    return LatentPromptCacheDataset(cached_latents, cached_prompts)

t2v_checkpoint_path = 'checkpoints/zeroscope_v2_576w'
# tokenizer = CLIPTokenizer.from_pretrained(t2v_checkpoint_path, subfolder='tokenizer')
# text_encoder = CLIPTextModel.from_pretrained(t2v_checkpoint_path, subfolder='text_encoder')
unet = UNet3DConditionModel.from_pretrained(t2v_checkpoint_path, subfolder='unet', torch_dtype=torch.float16).to('cpu')
# vae = AutoencoderKL.from_pretrained(t2v_checkpoint_path, subfolder='vae')
# scheduler = DDIMScheduler.from_pretrained(t2v_checkpoint_path, subfolder='scheduler')

for m_name, module in unet.named_modules():
    for p_name, parameter in module.named_parameter():
        print(f'parameter name: {p_name}')
     
    


lora_conf = LoraConfig(
    r=8, 
    lora_alpha=16, 
    target_modules=[
        'attn2', 
        ''
    ], 
    lora_dropout=0.1, 
    bias="none"
)

lora_unet = get_peft_model(unet, lora_conf)
