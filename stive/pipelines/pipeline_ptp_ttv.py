import os
import torch
import numpy as np
from typing import Union, List, Optional, Callable
from einops import rearrange
from peft import PeftModel
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers.models import UNet3DConditionModel, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from diffusers.pipelines import TextToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis import TextToVideoSDPipelineOutput
from ..prompt_attention import attention_util
from tqdm import trange, tqdm

class PtpTextToVideoSDPipeline(TextToVideoSDPipeline):
    def __init__(
        self, 
        text_encoder: CLIPTextModel, 
        tokenizer: CLIPTokenizer, 
        unet: UNet3DConditionModel | PeftModel, 
        vae: AutoencoderKL, 
        scheduler: DDIMScheduler, 
        disk_store: bool = False, 
    ):
        super().__init__(text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, vae=vae, scheduler=scheduler)
        self.disk_store = disk_store
        self.store_controller = attention_util.AttentionStore(disk_store=disk_store)
        self.empty_controller = attention_util.EmptyControl()
    
    @torch.no_grad()
    def prepare_ddim_source_latents(
        self, 
        frames, 
        text_embeddings,
        store_attention=False, 
        prompt=None,
        generator=None,
        LOW_RESOURCE=True,
        save_path = None
    ):
        self.prepare_before_train_loop()
        if store_attention:
            attention_util.register_attention_control(self, self.store_controller)
        resource_default_value = self.store_controller.LOW_RESOURCE
        self.store_controller.LOW_RESOURCE = LOW_RESOURCE  # in inversion, no CFG, record all latents attention
        b, f = frames.shape[:2]
        pixel_values = rearrange(frames, "b f h w c -> (b f) c h w", b=b)
        init_latents = self.vae.encode(pixel_values).latent_dist.sample(generator)
        print(f'self.vae.config.scaling_factor: {self.vae.config.scaling_factor}')
        init_latents = self.vae.config.scaling_factor * init_latents
        init_latents = rearrange(init_latents, "(b f) c h w -> b c f h w", b=b)
        
        ddim_latents_all_step = self.ddim_clean2noisy_loop(init_latents, text_embeddings, self.store_controller)
        if store_attention and (save_path is not None):
            os.makedirs(save_path+'/cross_attention', exist_ok=True)
            attention_output = attention_util.show_cross_attention(self.tokenizer, prompt, 
                                                                   self.store_controller, 16, ["up", "down"],
                                                                   save_path = save_path+'/cross_attention')

            # Detach the controller for safety
            attention_util.register_attention_control(self, self.empty_controller)
        self.store_controller.LOW_RESOURCE = resource_default_value
        
        return ddim_latents_all_step
    
    @torch.no_grad()
    def prepare_ddim_source_latents_with_latents(
        self, 
        init_latents, 
        text_embeddings,
        store_attention=False, 
        prompt=None,
        generator=None,
        LOW_RESOURCE=True,
        save_path = None
    ):
        self.prepare_before_train_loop()
        if store_attention:
            attention_util.register_attention_control(self, self.store_controller)
        resource_default_value = self.store_controller.LOW_RESOURCE
        self.store_controller.LOW_RESOURCE = LOW_RESOURCE  # in inversion, no CFG, record all latents attention
        b, f = init_latents.shape[:2]
        init_latents = rearrange(init_latents, "b f c h w -> b c f h w", b=b)
        
        ddim_latents_all_step = self.ddim_clean2noisy_loop(init_latents, text_embeddings, self.store_controller)
        if store_attention and (save_path is not None):
            os.makedirs(save_path+'/cross_attention', exist_ok=True)
            attention_output = attention_util.show_cross_attention(self.tokenizer, prompt, 
                                                                   self.store_controller, 16, ["up", "down"],
                                                                   save_path = save_path+'/cross_attention')

            # Detach the controller for safety
            attention_util.register_attention_control(self, self.empty_controller)
        self.store_controller.LOW_RESOURCE = resource_default_value
        
        return ddim_latents_all_step
    
    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latents, text_embeddings, controller:attention_util.AttentionControl=None):
        weight_dtype = latents.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        all_latents = [latents]
        latents = latents.clone().detach()
        for i in trange(len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            
            noise_pred = self.unet(latents, t, encoder_hidden_states=cond_embeddings).sample
            
            latents = self.next_clean2noise_step(noise_pred, t, latents)
            if controller is not None: controller.step_callback(latents)
            all_latents.append(latents.to(dtype=weight_dtype))
        
        return all_latents
    
    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    
    def ptp_replace_edit(
        self, 
        latents, 
        source_prompt, 
        target_prompt, 
        num_inference_steps, 
        is_replace_controller=True, 
        cross_replace_steps=None, 
        self_replace_steps=None, 
        blend_words=None, 
        equilizer_params=None, 
        use_inversion_attention=None, 
        blend_th=(0.3, 0.3), 
        fuse_th=0.3, 
        blend_self_attention=None, 
        blend_latents=None, 
        save_path=None, 
        save_self_attention=True, 
        guidance_scale=7.5, 
        generator=None, 
        disk_store=False, 
        cond_mask=None
    ):
        len_source = {len(source_prompt.split(' '))}
        len_target = {len(target_prompt.split(' '))}
        equal_length = (len_source == len_target)
        print(f"source prompt: {source_prompt}")
        print(f"target prompt: {target_prompt}")
        print(f"len_source: {len_source}, len_target: {len_target}, equal_length: {equal_length}")
        edit_controller = attention_util.make_controller(
                            self.tokenizer, 
                            [ source_prompt, target_prompt],
                            NUM_DDIM_STEPS = num_inference_steps,
                            is_replace_controller=is_replace_controller and equal_length,
                            cross_replace_steps=cross_replace_steps, 
                            self_replace_steps=self_replace_steps, 
                            blend_words=blend_words,
                            equilizer_params=equilizer_params,
                            additional_attention_store=self.store_controller,
                            use_inversion_attention=use_inversion_attention, 
                            blend_th=blend_th, 
                            fuse_th=fuse_th, 
                            blend_self_attention=blend_self_attention,
                            blend_latents=blend_latents,
                            save_path=save_path,
                            save_self_attention=save_self_attention,
                            disk_store=disk_store, 
                            cond_mask=cond_mask
                        )
        
        attention_util.register_attention_control(self, edit_controller)
        
        frames = self.ptp_replace_ddim(
            prompt=target_prompt, 
            latents=latents, 
            num_inference_steps=num_inference_steps, 
            controller=edit_controller, 
            guidance_scale=guidance_scale, 
            generator=generator, 
            ).frames
        
        
        if hasattr(edit_controller.latent_blend, 'mask_list'):
            mask_list = edit_controller.latent_blend.mask_list
        else:
            mask_list = None
        if len(edit_controller.attention_store.keys()) > 0:
            attention_output = attention_util.show_cross_attention(self.tokenizer, target_prompt, edit_controller, 16, ["up", "down"], save_path=save_path)
        else:
            attention_output = None
            
        dict_output = {
                "edited_frames" : frames,
                "attention_output" : attention_output,
                "mask_list" : mask_list,
            }
        attention_util.register_attention_control(self, self.empty_controller)
        del edit_controller
        return dict_output
        
    def prepare_before_train_loop(self, params_to_optimize=None):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        if params_to_optimize is not None:
            params_to_optimize.requires_grad = True
            
        
    @torch.no_grad()
    def ptp_replace_ddim(
        self,
        latents: torch.Tensor, 
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controller: attention_util.AttentionControl = None,
        **args
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, strength)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents_dtype = latents.dtype

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                print(f'latent_model_input: {latent_model_input.shape}')
                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample.to(dtype=latents_dtype)
                
                print(f'noise_pred: {noise_pred.shape}')
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # Edit the latents using attention map
                if controller is not None: 
                    dtype = latents.dtype
                    latents_new = controller.step_callback(latents)
                    latents = latents_new.to(dtype)
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                torch.cuda.empty_cache()
    
        # 8. Post processing
        if output_type == "latent":
            video = latents
        else:
            video_tensor = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video_tensor, output_type=output_type)
        
        if not return_dict:
            return (video,)

        return TextToVideoSDPipelineOutput(frames=video)

    def clear_store_controller(self):
        del self.store_controller
        self.store_controller = attention_util.AttentionStore(disk_store=self.disk_store)