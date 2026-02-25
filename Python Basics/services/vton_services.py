import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import AutoencoderKL, DDPMScheduler

from IDM_VTON.src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from IDM_VTON.src.unet_hacked_tryon import UNet2DConditionModel
from IDM_VTON.src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref

from controlnet_aux import OpenposeDetector


class VTONService:

    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = torch.float32 if self.device == "mps" else torch.float16

        model_id = "yisol/IDM-VTON"

        print("Loading IDM-VTON...")

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=self.dtype)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=self.dtype)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=self.dtype)
        self.unet_encoder = UNet2DConditionModel_ref.from_pretrained(model_id, subfolder="unet_encoder", torch_dtype=self.dtype)

        self.text_encoder_one = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=self.dtype)
        self.text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=self.dtype)

        self.tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
        self.tokenizer_two = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", use_fast=False)

        self.pipe = TryonPipeline.from_pretrained(
            model_id,
            unet=self.unet,
            vae=self.vae,
            feature_extractor=CLIPImageProcessor(),
            text_encoder=self.text_encoder_one,
            text_encoder_2=self.text_encoder_two,
            tokenizer=self.tokenizer_one,
            tokenizer_2=self.tokenizer_two,
            scheduler=self.noise_scheduler,
            image_encoder=self.image_encoder,
            unet_encoder=self.unet_encoder,
            torch_dtype=self.dtype,
        ).to(self.device)

        self.pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

        print("IDM-VTON Loaded Successfully")

    # -------------------------
    # Generate Pose Map
    # -------------------------
    def generate_pose(self, person_image: Image.Image):
        pose = self.pose_detector(person_image)
        return pose

    # -------------------------
    # Simple Torso Mask (can upgrade later)
    # -------------------------
    def generate_mask(self, image: Image.Image):
        w, h = image.size

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h*0.3):int(h*0.75), int(w*0.25):int(w*0.75)] = 255

        return Image.fromarray(mask)

    # -------------------------
    # Main Try-On Function
    # -------------------------
    def generate(self, person_image: Image.Image, cloth_image: Image.Image):

        person_image = person_image.resize((768, 1024))
        cloth_image = cloth_image.resize((768, 1024))

        pose_image = self.generate_pose(person_image)
        mask_image = self.generate_mask(person_image)

        prompt = "model is wearing the provided garment"
        negative_prompt = "low quality, distorted body, bad anatomy"

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_image,
            mask_image=mask_image,
            cloth=cloth_image,
            pose_map=pose_image,
            guidance_scale=2.5,
            num_inference_steps=30,
            height=1024,
            width=768,
        )

        return result.images[0]