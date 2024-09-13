<<<<<<< HEAD
=======
# Flux_Differential-diffusion_Inpainting_Diffusers
>>>>>>> 2202c2b73ceaa4b6a4c65e9da3c7227dd415d64e

 
## Introduction

This project implements soft inpainting using the Flux model with Diffusers, incorporating support for LoRA (Low-Rank Adaptation). It's based on several key technologies and techniques:

1. **Differential Diffusion**: A novel approach to image editing that allows for more precise and controllable modifications. This method is described in detail in the paper:

   [Differential Diffusion: Giving Each Pixel Its Strength](https://arxiv.org/abs/2306.00950)

   Differential Diffusion enables fine-grained control over the strength of diffusion at each pixel, leading to more nuanced and targeted image edits.

2. **Flux**: A state-of-the-art text-to-image model developed by Black Forest Labs. This project uses the development version of Flux, which can be found at:

   [FLUX.1-dev on Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-dev)

   Flux provides powerful image generation capabilities, which we leverage for our inpainting tasks.

3. **LoRA (Low-Rank Adaptation)**: This project supports LoRA for fine-tuning the Flux model. LoRA weights are loaded using the PEFT (Parameter-Efficient Fine-Tuning) method, allowing for efficient adaptation of the model to specific tasks or styles.

4. **Gaussian Blur for Mask Processing**: The user-provided mask undergoes Gaussian blur processing. This technique softens the edges of the mask, creating a more natural transition between the inpainted area and the original image.

By combining these technologies and techniques with the Diffusers library, this project offers a flexible and powerful tool for soft inpainting tasks, allowing for nuanced control over the inpainting process and easy adaptation to various styles or domains through LoRA.


 
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

4. Run the inpainting script:
   ```
   python inpaintingv3_Lora.py
   ```

5. The Gradio interface will launch in your default web browser. If it doesn't open automatically, look for a URL in the console output and open it manually.

6. In the Gradio interface:
   - Enter a prompt describing the image you want to generate
   - Upload an input image or use the drawing tool to create a mask
   - Adjust the sliders for inference steps, guidance scale, strength, and blur amount as needed
   - Click "Submit" to generate the inpainted image

7. The output image and the mask will be displayed in the interface. You can download them if desired.

Note: Make sure you have a GPU with sufficient VRAM for optimal performance. If you encounter memory issues, try reducing the image size or adjusting the model parameters.


## Advanced Usage: LoRA Support

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



<<<<<<< HEAD
=======


>>>>>>> 2202c2b73ceaa4b6a4c65e9da3c7227dd415d64e
