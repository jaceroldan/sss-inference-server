from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# SmolVLM - Instruct
from transformers import AutoProcessor, AutoModelForVision2Seq
# Qwen usage
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Make sure this is available

import torch
import cv2

import numpy as np

SMOLVLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
                                               torch_dtype=torch.bfloat16,
                                               _attn_implementation="flash_attention_2" if SMOLVLM_DEVICE == "cuda" else "eager").to(SMOLVLM_DEVICE)
model.to(SMOLVLM_DEVICE)

QWEN_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"" : 0}  # Force model to GPU
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# Request Schema
class InferenceRequest(BaseModel):
    prompt: str

# processing image
app = FastAPI()
@app.post("/caption_image")
async def caption_image(file: UploadFile = File(...)):
    print('RECEIVED!')
    # Read the uploaded frame
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform inference
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[frame], return_tensors="pt")
    inputs = inputs.to(SMOLVLM_DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts)
    return {
        'response': generated_texts
    }


@app.post("/decide_action")
async def predict(request: InferenceRequest, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Format input as per Qwen's expected structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]

        # Process inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the model's device
        inputs = {k: v.to(QWEN_DEVICE) for k, v in inputs.items()}

        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return {"response": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

