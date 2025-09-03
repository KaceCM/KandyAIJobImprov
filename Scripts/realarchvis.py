from dotenv import load_dotenv, find_dotenv
import os, torch, traceback
from time import time
from tqdm import tqdm
from utils import write_logs
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from PIL import Image

def to_rgb_depth(depth_pil: Image.Image, size=1024) -> Image.Image:
    d = depth_pil.convert("L").resize((size, size), Image.BICUBIC)
    return Image.merge("RGB", (d, d, d))

load_dotenv(find_dotenv(".env"))
MODEL_DIR = os.getenv("MODEL_DIR")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "./data/output_sdxl_realarchvis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sdxl_realarchvis():

    controlnet_model = "diffusers/controlnet-depth-sdxl-1.0"
    vae_model = "madebyollin/sdxl-vae-fp16-fix"
    dtype = torch.float16

    controlnet = ControlNetModel.from_pretrained(
        controlnet_model, torch_dtype=dtype, cache_dir=MODEL_DIR, variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(
        vae_model, torch_dtype=dtype, cache_dir=MODEL_DIR
    )

    base_ckpt_path = os.path.join(MODEL_DIR, "realarchvis_xlV30.safetensors")
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        base_ckpt_path,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
        cache_dir=MODEL_DIR,
    ).to(device)

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    return pipe, controlnet, vae

def apply_sdxl(pipe, prompt, negative_prompt, depth_rgb_1024, steps=28, guidance=7.0, cn_scale=0.8, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=depth_rgb_1024,
        controlnet_conditioning_scale=cn_scale,
        height=1024, width=1024,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return image

def main_sdxl_realarchvis_depthonly(image_list, verbose):
    pipe, controlnet, vae = load_sdxl_realarchvis()
    room_styles = ['scandinavian','minimalist','contemporary','parisian','industrial','rustic','japanese','art_deco','bohemian','coastal']


    try:
        for item in tqdm(image_list):
            for style in room_styles:
                try:
                    t0 = time()
                    name = item["name"]
                    depth = item["img"]

                    prompt = verbose["generation"][style]["living_room"]["positive"]
                    negative_prompt = verbose["generation"][style]["living_room"]["negative"]

                    # orig_w, orig_h = depth.size
                    depth_rgb = to_rgb_depth(depth, size=1024)

                    out = apply_sdxl(pipe, prompt, negative_prompt, depth_rgb)
                    # out_resized = out.resize((orig_w, orig_h), Image.LANCZOS)

                    out.save(os.path.join(OUTPUT_DIR, f"realarchvis_{name}_{style}.png"))
                    # out_resized.save(os.path.join(OUTPUT_DIR, f"realarchvis_{name}_{style}_resized.png"))
                    write_logs(f"[SDXL-RealArchvis] {name} ({style}) in {time()-t0:.2f}s")
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[SDXL-RealArchvis ERR] {name}/{style}: {e}")
                    write_logs(f"[SDXL-RealArchvis ERR] {name}/{style}: {e}\n{tb}")
                    continue
        del pipe
        del controlnet
        del vae
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[SDXL-RealArchvis ERR] General: {e}")
        write_logs(f"[SDXL-RealArchvis ERR] General: {e}\n{tb}")
    del pipe
    del controlnet
    del vae
