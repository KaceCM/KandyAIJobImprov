from Scripts.flux1d import main_flux_depthonly
from Scripts.sd35 import main_sd35_depthonly
from Scripts.sdxl import main_sdxl_depthonly
from Scripts.realarchvis import main_sdxl_realarchvis_depthonly
from download_checkpoint import download_realarchvis
import os
import json
import traceback
from diffusers.utils import load_image


def main():
    
    print("Starting processing...")

    print("--- Loading verbose.json ---")
    with open("verbose.json", "r") as f:
        verbose = json.load(f)

    print("--- Loading & Opening images from ./data/depth_map ---")
    image_in_dir = os.listdir("./data/depth_map")
    
    image_list = [{"name": img_name, "img": load_image(os.path.join("./data/depth_map", img_name)).convert("L")} for img_name in image_in_dir]
    print(f"Found {len(image_list)} images.")


    
    print("--- Starting generation with different models ---")

    print(">>> SDXL-RealArchvis generation <<<")
    try:
        print("--- Downloading checkpoints for RealArchVis ---")
        download_realarchvis()
        main_sdxl_realarchvis_depthonly(image_list, verbose)
    except Exception as e:
        tback = traceback.format_exc()
        print(f"Error in SDXL-RealArchvis: {e}\n{tback}")
        with open("log.txt", "a") as f:
            f.write(f"[SDXL-RealArchvis ERR] {e}\n{tback}\n")


    print(">>> FLUX.1 generation <<<")
    try:
        main_flux_depthonly(image_list, verbose)
    except Exception as e:
        tback = traceback.format_exc()
        print(f"Error in FLUX.1: {e}\n{tback}")
        with open("log.txt", "a") as f:
            f.write(f"[FLUX ERR] {e}\n{tback}\n")

    
    print(">>> SD3.5 generation <<<")    
    try:
        main_sd35_depthonly(image_list, verbose)
    except Exception as e:
        tback = traceback.format_exc()
        print(f"Error in SD3.5: {e}\n{tback}")
        with open("log.txt", "a") as f:
            f.write(f"[SD3.5 ERR] {e}\n{tback}\n")
    
    print(">>> SDXL generation <<<")
    try:
        main_sdxl_depthonly(image_list, verbose)
    except Exception as e:
        tback = traceback.format_exc()
        print(f"Error in SDXL: {e}\n{tback}")
        with open("log.txt", "a") as f:
            f.write(f"[SDXL ERR] {e}\n{tback}\n")

if __name__ == "__main__":
    main()
