from dotenv import load_dotenv, find_dotenv
import os, torch, traceback
from time import time
from tqdm import tqdm
from utils import write_logs
from diffusers import FluxControlNetPipeline, FluxControlNetModel

from PIL import Image
def to_rgb_depth(depth_pil: Image.Image, size=1024) -> Image.Image:
    d = depth_pil.convert("L").resize((size, size), Image.BICUBIC)
    return Image.merge("RGB", (d, d, d))

load_dotenv(find_dotenv(".env"))
MODEL_DIR = os.getenv("MODEL_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "./data/output_flux1d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_flux():
    base_model = "black-forest-labs/FLUX.1-dev"
    controlnet_model = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    controlnet = FluxControlNetModel.from_pretrained(
        controlnet_model, torch_dtype=dtype, cache_dir=MODEL_DIR
    )
    pipe = FluxControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=dtype, cache_dir=MODEL_DIR
    ).to(device)
    return pipe

def apply_flux(pipe, prompt, negative_prompt, depth_rgb_1024, steps=24, guidance=3.8, cn_scale=0.5, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_image=depth_rgb_1024,
        controlnet_conditioning_scale=cn_scale,
        width=1024, height=1024,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return image

def main_flux_depthonly(image_list, verbose):

    pipe = load_flux()
    room_styles = ['scandinavian','minimalist','contemporary','parisian','industrial','rustic','japanese','art_deco','bohemian','coastal']

    for item in tqdm(image_list):
        for style in room_styles:
            try:
                t0 = time()
                name = item["name"]
                depth = item["img"]

                prompt = verbose["generation"][style]["living_room"]["positive"]
                negative_prompt = verbose["generation"][style]["living_room"]["negative"]

                orig_w, orig_h = depth.size
                depth_rgb = to_rgb_depth(depth, size=1024)

                out = apply_flux(pipe, prompt, negative_prompt, depth_rgb)
                out_resized = out.resize((orig_w, orig_h), Image.LANCZOS)

                out.save(os.path.join(OUTPUT_DIR, f"flux1d_{name}_{style}.png"))
                out_resized.save(os.path.join(OUTPUT_DIR, f"flux1d_{name}_{style}_resized.png"))
                write_logs(f"[FLUX] {name} ({style}) in {time()-t0:.2f}s")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[FLUX ERR] {name}/{style}: {e}")
                write_logs(f"[FLUX ERR] {name}/{style}: {e}\n{tb}")
                continue
