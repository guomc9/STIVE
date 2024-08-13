# STIVE (Stable Textual Inversion Video Editing)

## Get started
### Prepare Environment
* To create environment with conda:
    ```shell
    conda env create --file environment.yml
    conda activate stive
    ```
* or setup environment with pip whether in a conda environment or not:
    ```shell
    pip3 install -r requirements.txt
    ```

### Download Pretrained Models
* SD1.4 model from: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main) (necessary)
* clip model from: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/main) (to evaluate clip and frame-consistency scores, not necessary)
* owlvit model from: [google/owlvit-base-patch16](https://huggingface.co/google/owlvit-base-patch16/tree/main) (to extract mask for attn-prob supervision)
* zeroscope model from: [cerspense/zeroscope_v2_576w](https://huggingface.co/cerspense/zeroscope_v2_576w/tree/main) (more efficent editing, recommend to download)

## Examples
<details>
<summary>Swap jeep to `$LAMBO`</summary>

* Swap jeep to `$LAMBO`
    - finetune concept from SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_lambo.yaml
        ```
    - finetune SD with spatial&temporal modules:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/jeep_to_lambo.yaml
        ```
    - prompt-to-prompt inference with concept and pretrained SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/lambo.yaml
        ```
    - prompt-to-prompt inference with concept and tuned SD:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/lambo.yaml
        ```
</details>
* Swap jeep to `$CYBERTRUCK`
    - finetune concept from SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_cybertruck.yaml
        ```
    - finetune SD with spatial&temporal modules:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/jeep_to_cybertruck.yaml
        ```
    - prompt-to-prompt inference with concept and pretrained SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/cybertruck.yaml
        ```
    - prompt-to-prompt inference with concept and tuned SD:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/cybertruck.yaml
        ```

* Swap jeep to `$FERRARI`
    - finetune concept from SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_ferrari.yaml
        ```
    - finetune SD with spatial&temporal modules:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/jeep_to_ferrari.yaml
        ```
    - prompt-to-prompt inference with concept and pretrained SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/ferrari.yaml
        ```
    - prompt-to-prompt inference with concept and tuned SD:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/ferrari.yaml
        ```

* Swap jeep to `$BMW`
    - finetune concept from SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_bmw.yaml
        ```
    - finetune SD with spatial&temporal modules:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/jeep_to_bmw.yaml
        ```
    - prompt-to-prompt inference with concept and pretrained SD: 
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/bmw.yaml
        ```
    - prompt-to-prompt inference with concept and tuned SD:
        ```shell
        CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/bmw.yaml
        ```
