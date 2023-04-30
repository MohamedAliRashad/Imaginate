import random
import re

import cv2
import einops
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from transformers import pipeline, set_seed

import config
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *
from pathlib import Path

gpt2_pipe = pipeline(
    "text-generation", model="Gustavosta/MagicPrompt-Stable-Diffusion", tokenizer="gpt2", device="cuda:0"
)

base_path = Path(__file__).parent

with open(base_path / "ideas.txt", "r") as f:
    line = f.readlines()


def generate_prompt(starting_text):
    seed = random.randint(100, 1000000)
    set_seed(seed)

    if starting_text == "":
        starting_text: str = line[random.randrange(0, len(line))].replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", "", starting_text)

    response = gpt2_pipe(
        starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4
    )
    response_list = []
    for x in response:
        resp = x["generated_text"].strip()
        if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
            return resp

    response_end = "\n".join(response_list)
    response_end = re.sub("[^ ]+\.[^ ]+", "", response_end)
    response_end = response_end.replace("<", "").replace(">", "")

    if response_end != "":
        return response_end


preprocessor = CannyDetector()

model_name = "control_v11p_sd15_canny"
model = create_model(base_path / f"models/{model_name}.yaml").cuda()
model.load_state_dict(load_state_dict(base_path / "models/deliberate_v2.ckpt", location="cuda"), strict=False)
model.load_state_dict(load_state_dict(base_path / f"models/{model_name}.pth", location="cuda"), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    detect_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    global preprocessor
    num_samples = int(num_samples)

    with torch.no_grad():
        input_image = HWC3(input_image)

        detected_map = preprocessor(resize_image(input_image, detect_resolution), low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return results

model_id = "XpucT/Deliberate"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda")

def run_deliberate(prompt, num_samples, image_resolution):
    num_samples = int(num_samples)
    prompt = [prompt] * num_samples
    images = pipe(prompt, height=image_resolution, width=image_resolution).images
    return images

with gr.Blocks(title="Imaginate", theme="sudeepshouche/minimalist").queue() as demo:
    gr.HTML("<center><h1> Image Generation using Diffusion Models </h1></center>")
    gr.HTML(
        "<center><h3>Enter image and initial prompt --> Infer cooler prompt --> Edit the input image using the prompt or generate a new image from the prompt.</h3></center>"
    )
    gr.HTML(
        "<center><p><b>Tip1:</b> Images get generated from <b>generated prompt</b> not from the initial prompt (you can edit them as you love)</p></center>"
    )
    gr.HTML(
        "<center><p><b>Tip2:</b> Use <a href='https://www.pinterest.com/'>pinterest</a> to get good initial images and don't forget tp play with the seed (Use `-1` if you want random seed with each run).</p></center>"
    )
    with gr.Row().style(equal_height=True):
        with gr.Column():
            input_image = gr.Image(source="upload", type="numpy")
            initial_prompt = gr.Textbox(label="Initial Prompt")
            generate_prompt_button = gr.Button(value="Generate Prompt")
            prompt = gr.Textbox(label="Generated Prompt", lines=3)
            run_button = gr.Button(value="Run Image Generation")
            run_only_prompt_button = gr.Button(value="Run Image Generation only from prompt")
            num_samples = gr.Radio(["1", "2", "4"], label="Number of Images to Generate", value="1")
            seed = gr.Number(label="Seed", value=6)
            with gr.Accordion("Advanced options", open=False):
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label="Guess Mode", value=False)
                detect_resolution = gr.Slider(
                    label="Preprocessor Resolution", minimum=128, maximum=1024, value=512, step=1
                )
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value="best quality")
                n_prompt = gr.Textbox(
                    label="Negative Prompt", value="lowres, bad anatomy, bad hands, cropped, worst quality"
                )
        with gr.Column():
            result_gallery = gr.Gallery(label="Generated Image").style(rows=2, columns=2, preview=True)

    examples = gr.Examples(
        [
            [
                "fa59914e1af9f7512722d7dba492ff7d.jpg",
                "Arab",
                "Arab, sword and shield, d & d, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, matte, sharp focus, illustration, hearthstone, art by artgerm and greg rutkowski and alphonse mucha",
                78410,
            ],
            [
                "337611cc2b73d3561fd614f6e2cb20aa.jpg",
                "Green",
                "Greenan Book Runner. Trending on Artstation, octane render, cinematic lighting from the right, hyper realism, octane render, 8k, depth of field, 3D",
                8685,
            ],
            [
                "e4a7c6b6d0235a4e67381b9e2a564962.jpg",
                "simple man",
                "simple man with black hair with a gray mustache. In style of Yoji Shinkawa and Hyung-tae Kim, trending on ArtStation, dark fantasy, great composition, concept art, highly detailed.",
                47510,
            ],
            [
                "7d26ef4a446181785cddd88b30bc2ef5.jpg",
                "Chinese",
                "Chinese art, fantasy, intricate, elegant, highly detailed, digital painting, artstation, concept art, matte, sharp focus, illustration, art by Artgerm and Greg Rutkowski and Alphonse Mucha",
                51354,
            ],
            [
                "9b9bfde243e309453cbf7b815c51e72e.jpg",
                "Realistic man holding sword in his hand",
                "Realistic man holding sword in his hand cinematic sci-fi art by Marc Simonetti and Greg Rutkowski, Ralph McQuarrie, James Gurney, artstation, cgsociety",
                50398,
            ],
        ],
        inputs=[input_image, initial_prompt, prompt, seed],
    )
    ips = [
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        detect_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
        low_threshold,
        high_threshold,
    ]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])
    run_only_prompt_button.click(fn=run_deliberate, inputs=[prompt, num_samples, image_resolution], outputs=[result_gallery])

    generate_prompt_button.click(fn=generate_prompt, inputs=[initial_prompt], outputs=[prompt])

demo.launch(
    share=True,
    enable_queue=True,
    favicon_path=base_path / "eco-bulb.png",
    show_api=False,
    height="100%",
)
