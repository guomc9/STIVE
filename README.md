# STIVE (Stable Textual Inversion Video Editing)

<!-- <video src="assets/jeep-unet-full-supvis/concat.mp4"> </video> -->
<video autoplay loop muted playsinline>
    <source src="assets/jeep-unet-full-supvis/concat.mp4" type="video/mp4">
    Your browser does not support the video tag.
</video>
<!-- ![examples](assets/jeep-unet-full-supvis/concat.gif) -->

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
* owlvit model from: [google/owlvit-base-patch16](https://huggingface.co/google/owlvit-base-patch16/tree/main) (to extract mask for attn-prob supervision, necessary for custom data)

## Examples
<details>
<summary>Swap jeep to <code>$LAMBO</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_lambo.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/car-turn/jeep_to_lambo.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/car-turn/lambo.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/car-turn/lambo.yaml</code></pre>
</li>
</ul>

</details>

<details>
<summary>Swap jeep to <code>$CYBERTRUCK</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_cybertruck.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/car-turn/jeep_to_cybertruck.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/car-turn/cybertruck.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/car-turn/cybertruck.yaml</code></pre>
</li>
</ul>

</details>

<details>
<summary>Swap jeep to <code>$FERRARI</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_ferrari.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/car-turn/jeep_to_ferrari.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/car-turn/ferrari.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/car-turn/ferrari.yaml</code></pre>
</li>
</ul>

</details>

<details>
<summary>Swap jeep to <code>$BMW</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_bmw.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/car-turn/jeep_to_bmw.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/car-turn/bmw.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/car-turn/bmw.yaml</code></pre>
</li>
</ul>

</details>


<details>
<summary>Swap jeep to <code>porsche</code></summary>
<ul>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/car-turn/jeep_to_porsche.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/car-turn/porsche.yaml</code></pre>
</li>
</ul>
</details>



<details>
<summary>Swap capsule to <code>$POKEBALL</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_pokeball.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/capsule-rot/capsule_to_pokeball.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/capsule-rot/pokeball.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/capsule-rot/pokeball.yaml</code></pre>
</li>

</ul>

</details>

<details>
<summary>Swap capsule to <code>$STAR</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_star.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/capsule-rot/capsule_to_star.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/capsule-rot/star.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/capsule-rot/star.yaml</code></pre>
</li>

</ul>

</details>


<!-- <details>
<summary>Swap flag to <code>$POKEFLAG</code></summary>
<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_pokeflag.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/usa-flag/flag_to_pokeflag.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/usa-flag/pokeflag.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/usa-flag/pokeflag.yaml</code></pre>
</li>
</ul>
</details>

<details>
<summary>Swap flag to <code>FrenchFlag</code></summary>
<ul>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/usa-flag/flag_to_frenchflag.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/usa-flag/frenchflag.yaml</code></pre>
</li>
</ul>
</details> -->


<details>
<summary>Swap lotus to <code>$CHRYSANTHEMUM</code></summary>

<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_chrysanthemum.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/lotus/lotus_to_chrysanthemum.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/lotus/chrysanthemum.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/lotus/chrysanthemum.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap man to <code>$NEO</code></summary>
<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_neo.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/man-skate/man_to_neo.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/man-skate/neo.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/man-skate/neo.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap man to <code>$OPTIMUS</code></summary>
<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_optimus.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/man-skate/man_to_optimus.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/man-skate/optimus.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/man-skate/optimus.yaml</code></pre>
</li>
</ul>
</details>

<details>
<summary>Swap man to <code>$SAVIOR</code></summary>
<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_savior.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/man-skate/man_to_savior.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/man-skate/savior.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/man-skate/savior.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap man to <code>spiderman</code></summary>
<ul>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/man-skate/man_to_spiderman.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/man-skate/spiderman.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap dog to <code>$CAT</code></summary>
<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_cat.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/kitten/kitten_to_cat.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/kitten/cat.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/kitten/cat.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap cat to <code>$FOX</code></summary>
<ul>
<li>finetune concept from SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/sd_concepts/sd_fox.yaml</code></pre>
</li>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/kitten/kitten_to_fox.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and pretrained SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/kitten/fox.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/kitten/fox.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap cat to <code>Shiba-Inu</code></summary>
<ul>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/kitten/kitten_to_Shiba-Inu.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/kitten/Shiba-Inu.yaml</code></pre>
</li>
</ul>
</details>


<details>
<summary>Swap cat to <code>tiger</code></summary>
<ul>
<li>finetune SD with spatial&amp;temporal modules:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/kitten/kitten_to_tiger.yaml</code></pre>
</li>
<li>prompt-to-prompt inference with concept and tuned SD:
    <pre><code>CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/kitten/tiger.yaml</code></pre>
</li>
</ul>
</details>