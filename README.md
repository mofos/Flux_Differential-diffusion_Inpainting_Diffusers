
# Flux_Differential-diffusion_Inpainting_Diffusers
can't wait for the PR(https://github.com/huggingface/diffusers/pull/9268) to get merge ..so used GPT to make this project..

## Introduction 

This project implements soft inpainting using the Flux model with Diffusers, incorporating support for LoRA (Low-Rank Adaptation). It's based on several key technologies and techniques:

there is inpaintingv3 with is simple inapinthing and inpaintingv3_Lora for inpinting with lora 

1. **inpaintingv3**:
   - Implements basic inpainting functionality.
   - Suitable for general image inpainting tasks.
   - Does not utilize LoRA (Low-Rank Adaptation) for fine-tuning.

2. **inpaintingv3_Lora**:
   - Extends the basic inpainting functionality with LoRA support.
   - Allows for fine-tuning the model using LoRA weights.
   - Provides more control and customization over the inpainting process.
   - Ideal for tasks requiring specific styles or adaptations.

 
## Installation and Usage Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Flux_Differential-diffusion_Inpainting_Diffusers.git
   cd Flux_Differential-diffusion_Inpainting_Diffusers
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the HF login to download flux dev script:
   ```
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login --token <hf_token>
   ```
5. Run the inpainting script:
   ```
   python inpaintingv3_Lora.py
   ```

6. The Gradio interface will launch in your default web browser. If it doesn't open automatically, look for a URL in the console output and open it manually.

7. In the Gradio interface:
   - Enter a prompt describing the image you want to generate
   - Upload an input image or use the drawing tool to create a mask
   - Adjust the sliders for inference steps, guidance scale, strength, and blur amount as needed
   - Click "Submit" to generate the inpainted image

8. The output image and the mask will be displayed in the interface. You can download them if desired.

Note: Make sure you have a GPU with sufficient VRAM for optimal performance. If you encounter memory issues, try reducing the image size or adjusting the model parameters.


## More details

1. **Differential Diffusion**:

   [Differential Diffusion: Giving Each Pixel Its Strength](https://arxiv.org/abs/2306.00950)

   Differential Diffusion enables fine-grained control over the strength of diffusion at each pixel, leading to more nuanced and targeted image edits.

2. **Flux**: 

   [FLUX.1-dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev)

   Flux provides powerful image generation capabilities, which we leverage for our inpainting tasks.

3. **LoRA (Low-Rank Adaptation)**: LoRA weights are loaded using the PEFT (Parameter-Efficient Fine-Tuning) method, allowing for efficient adaptation of the model to specific tasks or styles.

4. **Gaussian Blur for Mask Processing**: The user-provided mask undergoes Gaussian blur processing. This technique softens the edges of the mask, creating a more natural transition between the inpainted area and the original image.


This project now supports LoRA (Low-Rank Adaptation) for fine-tuning the Flux model. This feature is based on the work done in the Hugging Face Diffusers project, specifically PR #9268 (https://github.com/huggingface/diffusers/pull/9268). Credit goes to the contributors of that PR for implementing LoRA support in the Flux pipeline.

Here's how to use LoRA with this project:

1. Prepare your LoRA weights. You can either train your own or use pre-trained LoRA weights compatible with Flux.

2. Modify the `inpaintingv3_Lora.py` script to load and apply the LoRA weights. Here's an example of how to do this:

   ```python
   from diffusers import FluxDifferentialImg2ImgPipeline
   import torch

   # Load the base model
   pipe = FluxDifferentialImg2ImgPipeline.from_pretrained(
       "black-forest-labs/FLUX.1-dev",
       torch_dtype=torch.float16
   )

   # Load and fuse the LoRA weights
   pipe.load_lora_weights("path/to/your/lora/weights")

   # Set the LoRA scale (adjust as needed)
   pipe.set_adapters_scale(0.7)

   # Use the pipeline with LoRA
   output = pipe(
       prompt="Your prompt here",
       image=input_image,
       mask_image=mask_image,
       num_inference_steps=30,
       guidance_scale=7.5,
   ).images[0]
   ```

3. Adjust the LoRA scale (`set_adapters_scale`) to control the influence of the LoRA weights on the output. A higher value means stronger influence from the LoRA adaptation.

4. Experiment with different LoRA weights and scales to achieve the desired effect on your inpainting results.

Note: Using LoRA may require additional GPU memory. If you encounter out-of-memory errors, try reducing the image size or adjusting other parameters.

For more detailed information about the implementation and usage of LoRA in the Flux pipeline, please refer to the original PR: https://github.com/huggingface/diffusers/pull/9268


