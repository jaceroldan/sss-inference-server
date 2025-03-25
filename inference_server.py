def transform_agent(translate_x: float, translate_y: float, translate_z: float, degrees_x: float, degrees_y: float, degrees_z: float):
    """
    Transforms the agent by the specified translation and rotation with respect to the environment.
    
    Args:
        translate_x: Move the agent translate_x units to the right if this is positive, or to the left if negative.
        translate_y: Move the agent translate_y units up if this is positive, or down if negative.
        translate_z: Move the agent translate_z units forward if this is positive, or backward if negative.
        degrees_x: Rotate the agent's camera degrees_x units down if this is positive, or to up if negative.
        degrees_y: Rotate the agent's camera degrees_y units to the right if this is positive, or to the left if negative.
        degrees_z: Rotate agent's camera degrees_z units counterclockwise on its axis if this is positive, or clockwise if negative.
    Returns:
        The new position of the agent (x, y, z) with respect to the environment and its current camera rotation (x, y, z).
    """
    return 1

def transform_hands(
        left_translate_x: float, left_translate_y: float, left_translate_z: float,
        left_degrees_x: float, left_degrees_y: float, left_degrees_z: float,
        right_translate_x: float, right_translate_y: float, right_translate_z: float,
        right_degrees_x: float, right_degrees_y: float, right_degrees_z: float):
    """
    Transforms the agent's left hand and right hand by the specified translation and rotation with respect to the environment.
    
    Args:
        left_translate_x: Move the agent's left hand left_translate_x units to the right if this is positive, or to the left if negative.
        left_translate_y: Move the agent's left hand left_translate_y units up if this is positive, or down if negative.
        left_translate_z: Move the agent's left hand left_translate_z units forward if this is positive, or backward if negative.
        left_degrees_x: Rotate the agent's left hand rotation left_degrees_x units down if this is positive, or up if negative.
        left_degrees_y: Rotate the agent's left hand rotation left_degrees_y units to the right if this is positive, or to the left if negative.
        left_degrees_z: Rotate the agent's left hand rotation left_degrees_z units counterclockwise on its axis if this is positive, or clockwise if negative.
        right_translate_x: Move the agent's right hand right_translate_x units to the right if this is positive, or to the left if negative.
        right_translate_y: Move the agent's right hand right_translate_y units up if this is positive, or down if negative.
        right_translate_z: Move the agent's right hand right_translate_z units forward if this is positive, or backward if negative.
        right_degrees_x: Rotate the agent's right hand rotation right_degrees_x units down if this is positive, or up if negative.
        right_degrees_y: Rotate the agent's right hand rotation right_degrees_y units to the right if this is positive, or to the left if negative.
        right_degrees_z: Rotate the agent's right hand rotation right_degrees_z units counterclockwise on its axis if this is positive, or clockwise if negative.
    Returns:
        The new position of the agent's hands (x, y, z for both hands) with respect to the environment and both hands' current camera rotation (x, y, z for both).
    """
    return 1

def toggle_left_grip():
    """
    Toggles the grip of the left hand. Will successfully toggle from false to true if an XR Grab Interactable object collides with the left hand.

    Returns:
        True if an object has been grabbed and False, otherwise.
    """
    return 1


def toggle_right_grip():
    """
    Toggles the grip of the left hand. Will successfully toggle from false to true if an XR Grab Interactable object collides with the left hand.

    Returns:
        True if an object has been grabbed and False, otherwise.
    """
    return 1


tools = [transform_agent, transform_hands, toggle_left_grip, toggle_right_grip]


from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from io import BytesIO


# SmolVLM - Instruct
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers import AutoTokenizer, LlamaForCausalLM
# Qwen usage
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # Make sure this is available

import torch
import cv2

import numpy as np

SMOLVLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
smol_vlm_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
                                               torch_dtype=torch.bfloat16,
                                               _attn_implementation="flash_attention_2" if SMOLVLM_DEVICE == "cuda" else "eager").to(SMOLVLM_DEVICE)
smol_vlm_model.to(SMOLVLM_DEVICE)

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Pro-Llama-3-8B', trust_remote_code=True)

HERMES_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
hermes_model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
)

# processing image
app = FastAPI()
@app.post("/caption_image")
async def caption_image(prompt: str = Form(...), file: UploadFile = File(...)):
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
                {"type": "text", "text": prompt}
            ]
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[frame], return_tensors="pt")
    inputs = inputs.to(SMOLVLM_DEVICE)

    generated_ids = smol_vlm_model.generate(**inputs, max_new_tokens=256)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return {
        'response': generated_texts
    }


@app.post("/decide_action")
async def predict(prompt: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Format input as per Qwen's expected structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},  # Use PIL image
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(messages, chat_template="tool_use", tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(hermes_model.device) for k, v in inputs.items()}
        out = hermes_model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(out[0][len(inputs["input_ids"][0]):])
        print(response)

        return {"response": response}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

