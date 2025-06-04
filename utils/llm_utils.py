import json
from rich.console import Console
import os
import time
from PIL import Image
import torch
from huggingface_hub import hf_hub_download, login, list_repo_files
from src_inference.pipeline import FluxPipeline
from src_inference.lora_helper import set_single_lora
from utils.helper_utils import prepare_folder

from dotenv import load_dotenv
load_dotenv(override=True)

class AiArtOperation:
    def __init__(self):
        self.console = Console()
        self.converted_frames_folder = json.load(open('config/settings.json'))["convert_frame_folder"]
        self.proccessed_frames_folder = json.load(open('config/settings.json'))["proccessed_frame_folder"]
        self.ai_style = json.load(open('config/settings.json'))["ai_style"]
        self.base_model_path = json.load(open('config/settings.json'))["base_model_path"]
        self.hardware = json.load(open('config/settings.json'))["hardware"]
        self.cpu_offload = json.load(open('config/settings.json'))["cpu_offload"]
        self.prompt = json.load(open('config/settings.json'))["ai_art_prompt"]
        self.guidance_scale = json.load(open('config/settings.json'))["guidance_scale"]
        self.num_inference_steps = json.load(open('config/settings.json'))["num_inference_steps"]
        self.seed = json.load(open('config/settings.json'))["seed"]

        # preparing AI model
        self.console.print(f"*. AI Model Downloading...", style="bold blue")
        hf_hub_download(repo_id="showlab/OmniConsistency", filename=f"{self.ai_style}", local_dir="./LoRAs")
        self.omni_consistency_path = hf_hub_download(repo_id="showlab/OmniConsistency", filename="OmniConsistency.safetensors", local_dir="./Model")
        self.console.print(f"*. AI Model Downloading Complete!", style="bold blue")

        login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=False, new_session=True, write_permission=False)


    def clear_cache(self, transformer):
        for name, attn_processor in transformer.attn_processors.items():
            attn_processor.bank_kv.clear()


    def generate_image(self, pipe, custom_repo_id, prompt, input_img, width, height):
        width, height = int(width), int(height)
        generator = torch.Generator("cpu").manual_seed(self.seed)

        if custom_repo_id and custom_repo_id.strip():
            repo_id = custom_repo_id.strip()
            try:
                files = list_repo_files(repo_id)
                self.console.print(f"using custom LoRA from: {repo_id}", style="bold blue")
                safetensors_files = [f for f in files if f.endswith(".safetensors")]
                self.console.print(f"found safetensors files: {safetensors_files}", style="bold blue")
                if not safetensors_files:
                    raise ValueError("No .safetensors files were found in this repo")
                fname = safetensors_files[0]
                lora_path = hf_hub_download(repo_id=repo_id, filename=fname, local_dir="./Custom_LoRAs")
            except Exception as e:
                self.console.print(f"Load custom LoRA failed: {e}", style="bold red")
                exit()
        else:
            lora_path = os.path.join(
                f"./LoRAs/LoRAs", f"{self.ai_style}"
            )

        pipe.unload_lora_weights()
        try:
            pipe.load_lora_weights(
                os.path.dirname(lora_path),
                weight_name=os.path.basename(lora_path)
            )
        except Exception as e:
            self.console.print(f"Load LoRA failed: {e}", style="bold red")
            exit()

        spatial_image  = [input_img.convert("RGB")]
        subject_images = []
        start = time.time()
        out_img = pipe(
            prompt,
            height=(height // 8) * 8,
            width=(width  // 8) * 8,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            spatial_images=spatial_image,
            subject_images=subject_images,
            cond_size=512,
        ).images[0]
        self.console.print(f"inference time: {time.time()-start:.2f}s", style="bold blue")

        self.clear_cache(pipe.transformer)
        out_put_img_name = input_img.replace("\\", "/").rsplit("/", 1)[-1]
        prepare_folder(self.proccessed_frames_folder)
        out_img.save(f"{self.proccessed_frames_folder}/{out_put_img_name}")


    def convert_frames_to_ai_art(self):
        try:
            self.console.print(f"*. Preparing AI Pipeline..", style="bold blue")
            pipe = FluxPipeline.from_pretrained(self.base_model_path, torch_dtype=torch.bfloat16).to(self.hardware)
            set_single_lora(pipe.transformer, self.omni_consistency_path,lora_weights=[1], cond_size=512)
            if self.cpu_offload == True: pipe.enable_model_cpu_offload()

            image_files = [os.path.join(self.converted_frames_folder, f) for f in os.listdir(self.converted_frames_folder) if os.path.isfile(os.path.join(self.converted_frames_folder, f))]
            # Sort files by creation time (oldest to newest)
            image_files.sort(key=lambda x: os.path.getctime(x))
            for image in image_files:
                input_image = Image.open(image)
                width, height = input_image.size
                self.generate_image(pipe=pipe, prompt=self.prompt, input_img=input_image, width=width, height=height)

                
        except Exception as e:
            self.console.print(f"*. Error While proccessing AI Art: {e}", style="bold red")
            exit()