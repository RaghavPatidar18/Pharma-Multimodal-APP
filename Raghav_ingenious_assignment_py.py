# -*- coding: utf-8 -*-

!pip install gradio==3.44.4
!pip install diffusers==0.21.4
!pip install transformers==4.31.0

"""Make sure to Resatart the session , So that modules load properly"""

!pip install torch==2.0.1 torchvision==0.15.2
!pip install accelerate==0.22.0
!pip install safetensors
!pip install huggingface_hub==0.15.1
!pip install opencv-python==4.8.0.76

"""Make sure to Resatart the session , So that modules load properly

# Follow Below steps to run the first cell , i.e Hugginface Login

*   Please get the Hugging Face login token : 
*   Create a Secret Key as : HF_TOKEN and give value as above token and allow access
"""

from huggingface_hub import login
login()   # it requires a token

import gradio as gr
import torch
import gc
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initializing global pipeline variables
text2img_pipe = None
img2img_pipe = None
controlnet_pipe = None

def unload_pipeline():
    gc.collect()
    torch.cuda.empty_cache()

# It Loads all the required pipelines at app startup
def load_pipelines_on_startup():
    global text2img_pipe, img2img_pipe , controlnet_pipe
    print("[INFO] Loading pipelines at startup...")

    # Load the Text-to-Image Stable Diffusion pipeline
    if text2img_pipe is None:
        text2img_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16           # Use fp16 for faster inference
        ).to(device)

    # Load the Image-to-Image Stable Diffusion pipeline
    if img2img_pipe is None:
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to(device)

    # Load ControlNet pipeline with Canny edge detection
    if controlnet_pipe is None:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16  # Canny ControlNet variant
        ).to(device)

        controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            controlnet=controlnet,    # Injected ControlNet into the pipeline
            torch_dtype=torch.float16
        ).to(device)

        # Used a more advanced scheduler for better quality/speed tradeoff
        controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(controlnet_pipe.scheduler.config)

    print("[INFO] Pipelines loaded.")


# It convert an image into a canny edge map for use with ControlNet
def get_canny_image(image, low_threshold=100, high_threshold=200):
    image = np.array(image.convert("RGB"))
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = np.stack([image]*3, axis=-1)
    return Image.fromarray(image)

# Function generate multiple images using the Text-to-Image pipeline
def generate_multiple_images(prompt, num_images=3):
    try:
        print(f"[INFO] Generating {num_images} images for prompt: {prompt}")

        # Maintaining pharma context in prompt
        domain_suffix = (
            "in a pharmaceutical setting, possibly with lab equipment, bioreactors, or cleanroom background"
        )

        full_prompt = f"{prompt}, {domain_suffix}"
        pipe = text2img_pipe
        results = [pipe(full_prompt).images[0] for _ in range(num_images)] # it generates 3 images
        return results , results # Return twice (for UI use with gallery + state)

    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        # Return blank images in case of error
        return [Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)) for _ in range(num_images)]


# This function regenerate image based on edited prompt and previously selected image
def regenerate_image(selected_img,index, new_prompt):
    try:
        index = int(index)
        selected_img = selected_img[index]

        if selected_img is None:
            raise ValueError("No image selected for regeneration.")

        # Maintaining pharma context in prompt
        domain_suffix = (
            "in a pharmaceutical setting, possibly with lab equipment, bioreactors, or cleanroom background"
        )
        full_prompt = f"{new_prompt}, {domain_suffix}"
        pipe = img2img_pipe  # our global variable pipeline
        img = pipe(prompt=full_prompt, image=selected_img, strength=0.75).images[0]
        return img

    except Exception as e:
        print(f"[ERROR] Regeneration failed: {e}")
        # Return blank image on failure
        return Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))


# This function generate multiple images from a user-provided image and prompt using Img2Img
def generate_from_image(prompt, image , num_images=3):
    try:
        # Maintaining pharma context in prompt
        domain_suffix = (
            "in a pharmaceutical setting, possibly with lab equipment, bioreactors, or cleanroom background"
        )
        full_prompt = f"{prompt}, {domain_suffix}"
        pipe = img2img_pipe
        image = image.resize((512, 512)) # Resize input to 512x512 for stable diffusion
        img = [pipe(prompt=full_prompt, image=image, strength=0.75).images[0] for _ in range(num_images)]
        return img ,img # Output for gallery and state

    except Exception as e:
        print(f"[ERROR] Img2Img failed: {e}")
        return Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))


# This function store selected image from Gradio gallery
def store_selected(index, gallery_images):
    if index is None or gallery_images is None:
        raise gr.Error("No image was selected.")
    selected_img = gallery_images[index]  # Select image from gallery based on user click
    return selected_img, selected_img


# Generate an image using ControlNet (Canny) based on prompt and input image
def generate_controlnet_image(prompt, input_image):
    try:
        canny_image = get_canny_image(input_image)   # Getting Canny edge map from input image
        pipe = controlnet_pipe
        result = pipe(prompt=prompt, image=canny_image, num_inference_steps=20).images[0]  # It generate image using ControlNet
        return result
    except Exception as e:
        print(f"[ERROR] ControlNet generation failed: {e}")
        return Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))

with gr.Blocks() as demo:

# ------------------------ TAB 1: TEXT TO IMAGE ------------------------

    with gr.Tab("Text to Image"):
        prompt = gr.Textbox(label="Prompt", lines=2)  # promt input
        gen_btn = gr.Button("Generate")

        gallery = gr.Gallery(label="Choose one").style(grid=[3], height="auto")  # Show outputs
        imgs = gr.State()   # State to store all generated images
        selected = gr.Number(show_label=False)

        # Function to update selected image index when user clicks on a gallery image
        def update_selected(evt: gr.SelectData):
            return evt.index

        gallery.select(update_selected,None,selected)  # Display for regenerated image

        # Edit prompt UI: input + button
        with gr.Row():
            edit_prompt = gr.Textbox(label="Edit Prompt")
            edit_btn = gr.Button("Generate Edited Image")

        edited_output = gr.Image(label="Edited Image")

        gen_btn.click(generate_multiple_images, inputs=prompt, outputs=[gallery,imgs])
        edit_btn.click(regenerate_image, inputs=[imgs, selected, edit_prompt], outputs=edited_output)

# ------------------------ TAB 2: IMAGE TO IMAGE ------------------------

    with gr.Tab("Image to Image"):
        img2img_prompt = gr.Textbox(label="Prompt")  # promt input
        init_image = gr.Image(type="pil", label="Initial Image")  # input image
        gen_btn = gr.Button("Generate")

        gallery = gr.Gallery(label="Choose one").style(grid=[3], height="auto")
        imgs = gr.State()   # State to store all generated images
        selected = gr.Number(show_label=False)  # Track selected index

        def update_selected(evt: gr.SelectData):
            return evt.index

        gallery.select(update_selected,None,selected)

        # Edit functionality: input + button
        with gr.Row():
            edit_prompt = gr.Textbox(label="Edit Prompt")
            edit_btn = gr.Button("Generate Edited Image")

        edited_output = gr.Image(label="Edited Image")  # Display for regenerated image

        gen_btn.click(generate_from_image, inputs=[img2img_prompt, init_image], outputs=[gallery,imgs])
        edit_btn.click(regenerate_image, inputs=[imgs, selected, edit_prompt], outputs=edited_output)

# ------------------------ TAB 3: CONTROLNET IMAGE ENHANCEMENT ------------------------

    with gr.Tab("Enhance Your Image"):
            cn_prompt = gr.Textbox(label="Prompt")  # Prompt for ControlNet-based generation
            cn_input = gr.Image(type="pil", label="Input Image (for edge detection)")  # Upload image for ControlNet
            cn_generate = gr.Button("Generate with ControlNet")
            cn_output = gr.Image(label="Generated Output")  # Display generated image

            cn_generate.click(generate_controlnet_image, inputs=[cn_prompt, cn_input], outputs=cn_output) # Connect ControlNet button

if __name__ == "__main__":
    load_pipelines_on_startup()   # loading all the pipe lines at the start only
    demo.launch(share=True, debug=True)