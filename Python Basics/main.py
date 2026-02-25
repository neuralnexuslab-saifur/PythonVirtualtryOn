from fastapi import FastAPI , UploadFile , File
from pydantic import BaseModel
from transformers import pipeline,SamModel, SamProcessor ,CLIPProcessor, CLIPModel
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import torch
import cv2
import numpy as np

from services.vton_services import VTONService
import io
import os

vton_service = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

@app.on_event("startup")
def load_model():
    global vton_service
    vton_service = VTONService()
# Load model once (important!)
classifier = pipeline("image-classification")

class Item(BaseModel):
    name:str
    price:float
    isOffer:bool = None
@app.get("/")
def read_root():
    return {
    "message": "Hello World"
}

@app.get("/items/{item_id}")
def read_item(item_id:int,q:str):
    return {"item_id":item_id,"query":q }

person_path = os.path.join(BASE_DIR, "person.png")
mask_path = os.path.join(BASE_DIR, "test_data/test/agnostic_mask/person_mask.png")
cloth_path = os.path.join(BASE_DIR, "test_data/test/cloth/person.png")
device =  "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

def get_cloth_embedding(cloth_image):
    inputs = clip_processor(images=cloth_image, return_tensors="pt").to(device)
    outputs = clip_model.get_image_features(**inputs)
    return outputs

def generate_mask(person_image):
    inputs = sam_processor(person_image, return_tensors="pt").to(device)
    outputs = sam_model(**inputs)

    masks = outputs.pred_masks.squeeze().detach().cpu().numpy()

    # Take largest mask (simplified for PoC)
    mask = masks[0]
    mask = (mask > 0.5).astype("uint8") * 255

    return Image.fromarray(mask)

def createMask():
    image_path = os.path.join(BASE_DIR, "test_data", "test", "image", "person.jpg")
    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    x1 = int(width * 0.25)
    y1 = int(height * 0.25)
    x2 = int(width * 0.75)
    y2 = int(height * 0.75)

    input_boxes = [[[x1, y1, x2, y2]]]

    inputs = sam_processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt"
    )

    # ✅ Fix for MPS
    inputs = sam_processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt"
    )

    # Move to device safely
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(device, dtype=torch.float32)
        else:
            inputs[k] = v.to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    mask_tensor = masks[0][0][0]  # batch=0, mask=0
    mask = mask_tensor.cpu().numpy()

    mask = (mask > 0).astype(np.uint8) * 255

    # Ensure it's 2D
    mask = mask.squeeze()

    print("Mask shape:", mask.shape)  # should print (H, W)

    mask_image = Image.fromarray(mask, mode="L")
    mask_image.save("clothing_mask.png")

    return mask_image
@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    print("updating item")
    global vton_service
    createdMask = createMask()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float32
    ).to(device)

    pipe.enable_attention_slicing()

    person = Image.open("test_data/test/image/person.jpg").convert("RGB").resize((512, 512))
    cloth = Image.open("test_data/test/cloth/person.png").convert("RGB")

    # Auto mask
    # mask = generate_mask(person)
    #
    # # Extract cloth embedding
    # cloth_embedding = get_cloth_embedding(cloth)
    #
    # prompt = """
    #    A realistic photo of the same person wearing the provided clothing,
    #    preserve exact texture, fabric folds, and identity.
    #    """
    #
    # negative_prompt = "different person, distorted face, blurry"
    #
    # result = pipe(
    #     prompt=prompt,
    #     negative_prompt=negative_prompt,
    #     image=person,
    #     mask_image=mask,
    #     guidance_scale=7.5,
    #     num_inference_steps=30
    # )
    #
    # result.images[0].save("tryon_output.png")
    output = vton_service.generate(person, cloth)

    output.save("result.png")
    return {
        "item_id": item_id,
        "item_name": "saved output.png",
        "item_price": item.price
    }



@app.post("/items")
async def read_item(image: UploadFile = File(...)):
    contents = await image.read()

    img = Image.open(io.BytesIO(contents))

    result = classifier(img)

    return {
        "filename": image.filename,
        "content_type": image.content_type,
        "description":result
    }















