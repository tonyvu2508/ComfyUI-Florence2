from collections.abc import Callable
import gc
import torch
import torchvision.transforms.functional as F
import io
import os
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageColor, ImageFont
import random
import numpy as np
import re
from pathlib import Path

#workaround for unnecessary flash_attn requirement
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    try:
        if not str(filename).endswith("modeling_florence2.py"):
            return get_imports(filename)
        imports = get_imports(filename)
        imports.remove("flash_attn")
    except:
        print(f"No flash_attn import to remove")
        pass
    return imports


def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]): 
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """

    flattened_paths = [item for path in paths if Path(path).exists() for item in Path(path).iterdir() if predicate(item)]

    return {item.name: str(item.absolute()) for item in flattened_paths}


import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(model_directory, exist_ok=True)

# Ensure ComfyUI knows about the LLM model path
folder_paths.add_model_folder_path("LLM", model_directory)

from transformers import AutoModelForCausalLM, AutoProcessor, set_seed

class DownloadAndLoadFlorence2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'microsoft/Florence-2-base',
                    'microsoft/Florence-2-base-ft',
                    'microsoft/Florence-2-large',
                    'microsoft/Florence-2-large-ft',
                    'HuggingFaceM4/Florence-2-DocVQA',
                    'thwri/CogFlorence-2.1-Large',
                    'thwri/CogFlorence-2.2-Large',
                    'gokaygokay/Florence-2-SD3-Captioner',
                    'gokaygokay/Florence-2-Flux-Large',
                    'MiaoshouAI/Florence-2-base-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-large-PromptGen-v1.5',
                    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
                    'MiaoshouAI/Florence-2-large-PromptGen-v2.0'
                    ],
                    {
                    "default": 'microsoft/Florence-2-base'
                    }),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Florence2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        print(f"Florence2 using {attention} for attention")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device, torch_dtype=dtype,trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)
        
        florence2_model = {
            'model': model, 
            'processor': processor,
            'dtype': dtype
            }

        return (florence2_model,)
    
class DownloadAndLoadFlorence2Lora:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'NikshepShetty/Florence-2-pixelprose',
                    ],
                  ),            
            },
          
        }

    RETURN_TYPES = ("PEFTLORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model):
        model_name = model.rsplit('/', 1)[-1]
        model_path = os.path.join(model_directory, model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading Florence2 lora model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
        return (model_path,)
    
class Florence2ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        all_llm_paths = folder_paths.get_folder_paths("LLM")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())

        return {"required": {
            "model": ([*s.model_paths], {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
            "precision": (['fp16','bf16','fp32'],),
            "attention": (
                    [ 'flash_attention_2', 'sdpa', 'eager'],
                    {
                    "default": 'sdpa'
                    }),
            },
            "optional": {
                "lora": ("PEFTLORA",),
            }
        }

    RETURN_TYPES = ("FL2MODEL",)
    RETURN_NAMES = ("florence2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Florence2"

    def loadmodel(self, model, precision, attention, lora=None):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = Florence2ModelLoader.model_paths.get(model)
        print(f"Loading model from {model_path}")
        print(f"Florence2 using {attention} for attention")
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation=attention, device_map=device, torch_dtype=dtype,trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if lora is not None:
            from peft import PeftModel
            adapter_name = lora
            model = PeftModel.from_pretrained(model, adapter_name, trust_remote_code=True)
        
        florence2_model = {
            'model': model, 
            'processor': processor,
            'dtype': dtype
            }
   
        return (florence2_model,)
    
class Florence2Run:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "florence2_model": ("FL2MODEL", ),
                "text_input": ("STRING", {"default": "", "multiline": True}),
                "task": (
                    [ 
                    'region_caption',
                    'dense_region_caption',
                    'region_proposal',
                    'caption',
                    'detailed_caption',
                    'more_detailed_caption',
                    'caption_to_phrase_grounding',
                    'referring_expression_segmentation',
                    'ocr',
                    'ocr_with_region',
                    'docvqa',
                    'prompt_gen_tags',
                    'prompt_gen_mixed_caption',
                    'prompt_gen_analyze',
                    'prompt_gen_mixed_caption_plus',
                    ],
                   ),
                "fill_mask": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "output_mask_select": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "JSON")
    RETURN_NAMES =("image", "mask", "caption", "data") 
    FUNCTION = "encode"
    CATEGORY = "Florence2"

    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)

    def encode(self, image, text_input, florence2_model, task, fill_mask, keep_model_loaded=False, 
           num_beams=3, max_new_tokens=1024, do_sample=True, output_mask_select="", seed=None):

        device = mm.get_torch_device()
        _, height, width, _ = image.shape
        offload_device = mm.unet_offload_device()

        processor = florence2_model['processor']
        model = florence2_model['model']
        dtype = florence2_model['dtype']

        if seed:
            set_seed(self.hash_seed(seed))

        colormap = ['blue','orange','green','purple','brown','pink','olive','cyan','red',
                    'lime','indigo','violet','aqua','magenta','gold','tan','skyblue']

        prompts = {
            'region_caption': '<OD>',
            'dense_region_caption': '<DENSE_REGION_CAPTION>',
            'region_proposal': '<REGION_PROPOSAL>',
            'caption': '<CAPTION>',
            'detailed_caption': '<DETAILED_CAPTION>',
            'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
            'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
            'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
            'docvqa': '<DocVQA>',
            'prompt_gen_tags': '<GENERATE_TAGS>',
            'prompt_gen_mixed_caption': '<MIXED_CAPTION>',
            'prompt_gen_analyze': '<ANALYZE>',
            'prompt_gen_mixed_caption_plus': '<MIXED_CAPTION_PLUS>',
        }
        task_prompt = prompts.get(task, '<OD>')

        if (task not in ['referring_expression_segmentation', 'caption_to_phrase_grounding', 'docvqa']) and text_input:
            raise ValueError("Text input (prompt) is only supported for 'referring_expression_segmentation', 'caption_to_phrase_grounding', and 'docvqa'")

        prompt = f"{task_prompt} {text_input}" if text_input else task_prompt
        image = image.permute(0, 3, 1, 2)

        out, out_masks, out_results, out_data = [], [], [], []
        pbar = ProgressBar(len(image))

        model.to(device)

        with torch.no_grad():
            for img in image:
                image_pil = F.to_pil_image(img)
                inputs = processor(text=prompt, images=image_pil, return_tensors="pt", do_rescale=False).to(dtype).to(device)

                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                )
                results = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                print(results)
                del generated_ids
                torch.cuda.empty_cache()
                gc.collect()

                if task == 'ocr_with_region':
                    cleaned_string = re.sub(r'</?s>|<[^>]*>', '\n', str(results))
                    clean_results = re.sub(r'\n+', '\n', cleaned_string)
                else:
                    clean_results = str(results).replace('</s>', '').replace('<s>', '')

                if len(image) == 1:
                    out_results = clean_results
                else:
                    out_results.append(clean_results)

                W, H = image_pil.size
                parsed_answer = processor.post_process_generation(results, task=task_prompt, image_size=(W, H))

                if task in ['region_caption', 'dense_region_caption', 'caption_to_phrase_grounding', 'region_proposal']:
                    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
                    ax.imshow(image_pil)
                    bboxes = parsed_answer[task_prompt]['bboxes']
                    labels = parsed_answer[task_prompt]['labels']
                    mask_indexes = output_mask_select.split(",") if output_mask_select else [str(i) for i in range(len(bboxes))]

                    if fill_mask:
                        mask_layer = Image.new('RGB', image_pil.size, (0, 0, 0))
                        mask_draw = ImageDraw.Draw(mask_layer)

                    for index, (bbox, label) in enumerate(zip(bboxes, labels)):
                        indexed_label = f"{index}.{label}"
                        if fill_mask and (str(index) in mask_indexes or label in mask_indexes):
                            mask_draw.rectangle(bbox, fill=(255, 255, 255))

                        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                                linewidth=1, edgecolor='r', facecolor='none', label=indexed_label)
                        ax.add_patch(rect)

                        text_x, text_y = bbox[0], bbox[1] - 12
                        text_x = max(0, min(text_x, W - len(label) * 6))
                        text_y = text_y if text_y >= 0 else bbox[3]

                        ax.text(text_x, text_y, indexed_label, color='white', fontsize=12,
                                bbox=dict(facecolor=random.choice(colormap) if len(image) == 1 else 'red', alpha=0.5))

                    if fill_mask:
                        mask_tensor = F.to_tensor(mask_layer).unsqueeze(0).permute(0, 2, 3, 1).cpu().float().mean(dim=0, keepdim=True)
                        out_masks.append(mask_tensor.repeat(1, 1, 1, 3)[:, :, :, 0])

                    ax.axis('off')
                    fig.canvas.draw()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', pad_inches=0)
                    buf.seek(0)
                    annotated_image_pil = Image.open(buf)
                    out_tensor = F.to_tensor(annotated_image_pil)[:3, :, :].unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
                    out.append(out_tensor)
                    out_data.append(parsed_answer[task_prompt] if task == 'caption_to_phrase_grounding' else bboxes)
                    plt.close(fig)

                pbar.update(1)

                del inputs, results, clean_results, parsed_answer
                torch.cuda.empty_cache()
                gc.collect()

        out_tensor = torch.cat(out, dim=0) if out else torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
        out_mask_tensor = torch.cat(out_masks, dim=0) if out_masks else torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")

        if not keep_model_loaded:
            print("Offloading model...")
            model.to(offload_device)
            mm.soft_empty_cache()

        return out_tensor, out_mask_tensor, out_results, out_data


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": DownloadAndLoadFlorence2Model,
    "DownloadAndLoadFlorence2Lora": DownloadAndLoadFlorence2Lora,
    "Florence2ModelLoader": Florence2ModelLoader,
    "Florence2Run": Florence2Run,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFlorence2Model": "DownloadAndLoadFlorence2Model",
    "DownloadAndLoadFlorence2Lora": "DownloadAndLoadFlorence2Lora",
    "Florence2ModelLoader": "Florence2ModelLoader",
    "Florence2Run": "Florence2Run",
}
