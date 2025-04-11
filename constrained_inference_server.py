def move_forward():
    """
    Moves the agent 0.1 units forward.

    Returns:
        The amount of movement by the agent in the forward direction.
    """
    return 0.1


def move_backward():
    """
    Moves the agent 0.1 units backward.

    Returns:
        The amount of movement by the agent in the backward direction.
    """
    return 0.1


def move_left():
    """
    Moves the agent 0.1 units to the left.

    Returns:
        The amount of movement by the agent in the leftward direction.
    """
    return 0.1


def move_right():
    """
    Moves the agent 0.1 units to the right.

    Returns:
        The amount of movement by the agent in the rightward direction.
    """
    return 0.1


def pan_left():
    """
    Pans the camera's agent to look 2.5 degrees to the left.

    Returns:
        The amount of camera panning by the agent in the right-to-left orientation.
    """
    return 2.5


def pan_right():
    """
    Pans the camera's agent to look 2.5 degrees to the right.

    Returns:
        The amount of camera panning by the agent in the left-to-right orientation.
    """
    return 2.5


def pan_up():
    """
    Pans the camera's agent to look 2.5 degrees upwards.

    Returns:
        The amount of camera panning by the agent in the down-to-up orientation.
    """
    return 2.5


def pan_down():
    """
    Pans the camera's agent to look 2.5 degrees downwards.

    Returns:
        The amount of camera panning by the agent in the up-to-down orientation.
    """
    return 2.5


def extend_left_hand_forward():
    """
    Extends the agent's left hand forward by 0.025 units.

    Returns:
        The amount of movement by the left hand in the forward direction.
    """
    return 0.025


def extend_right_hand_forward():
    """
    Extends the agent's right hand forward by 0.025 units.

    Returns:
        The amount of movement by the right hand in the forward direction.
    """
    return 0.025


def pull_left_hand_backward():
    """
    Pulls the agent's left hand backward by 0.025 units.

    Returns:
        The amount of movement by the left hand in the backward direction.
    """
    return 0.025


def pull_right_hand_backward():
    """
    Pulls the agent's right hand backward by 0.025 units.

    Returns:
        The amount of movement by the right hand in the backward directino.
    """
    return 0.025


def raise_left_hand():
    """
    Raises the agent's left hand by 0.025 units.

    Returns:
        The amount of movement by the left hand in the down-to-up direction.
    """
    return 0.025


def raise_right_hand():
    """
    Raises the agent's right hand by 0.025 units.

    Returns:
        The amount of movement by the right hand in the down-to-up direction.
    """
    return 0.025


def lower_left_hand():
    """
    Lowers the agent's left hand by 0.025 units.

    Returns:
        The amount of movement by the left hand in the up-to-down direction.
    """
    return 0.025


def lower_right_hand():
    """
    Lowers the agent's right hand by 0.025 units.

    Returns:
        The amount of movement by the right hand in the up-to-down direction.
    """
    return 0.025


def toggle_left_grip():
    """
    Toggles the grip of the left hand.

    Returns:
        True if an object has been grabbed and False, otherwise.
    """
    return True


def toggle_right_grip():
    """
    Toggles the grip of the right hand.

    Returns:
        True if an object has been grabbed and False, otherwise.
    """
    return True


tools = [
    move_forward,
    move_backward,
    move_left,
    move_right,
    pan_left,
    pan_right,
    pan_up,
    pan_down,
    extend_left_hand_forward,
    extend_right_hand_forward,
    pull_left_hand_backward,
    pull_right_hand_backward,
    raise_left_hand,
    raise_right_hand,
    lower_left_hand,
    lower_right_hand,
    toggle_left_grip,
    toggle_right_grip
]


from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from io import BytesIO


# SmolVLM - Instruct
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
# Hermes
from transformers import AutoTokenizer, LlamaForCausalLM

import torch
import cv2

import numpy as np
############################ 
# SMOLVLM
############################
SMOLVLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Commented out to reduce memory usage
# processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
# smol_vlm_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct",
#                                                torch_dtype=torch.bfloat16,
#                                                _attn_implementation="flash_attention_2" if SMOLVLM_DEVICE == "cuda" else "eager")
# smol_vlm_model.to(SMOLVLM_DEVICE)

processor_light = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
smol_vlm_model_light = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda")
smol_vlm_model_light.to(SMOLVLM_DEVICE)


############################ 
# Hermes
############################
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Pro-Llama-3-8B', trust_remote_code=True)

HERMES_DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"
hermes_model = LlamaForCausalLM.from_pretrained(
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
)

app = FastAPI()

# processing images
@app.post("/caption_image_light")
async def caption_image(prompt: str = Form(...), file: UploadFile = File(...)):
    print('RECEIVED!')
    # Read the uploaded frame
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert OpenCV BGR image to PIL RGB image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Perform inference
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    # Step 1: Format the chat template
    prompt_text = processor_light.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False  # <-- Make sure to keep tokenize=False here
    )

    # Step 2: Tokenize + encode the inputs correctly with images
    inputs = processor_light(
        text=prompt_text,
        images=[pil_image],
        return_tensors="pt"
    ).to(SMOLVLM_DEVICE)

    # Optional: Adjust dtype if using bfloat16
    if smol_vlm_model_light.dtype == torch.bfloat16:
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

    # Step 3: Generate
    with torch.no_grad():
        generated_ids = smol_vlm_model_light.generate(**inputs, max_new_tokens=256)
        generated_texts = processor_light.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

    return {
        'response': generated_texts
    }


# Verbose set of steps to achieve a goal.
@app.post("/plan_actions")
async def plan_actions(goal: str = Form(...), caption: str = Form(...)):
    """
    Generates a strict sequence of steps based on the goal and environment description.
    """
    prompt = f"""
    You are an AI planning agent for a virtual grocery environment.
    Your task is to break down the following goal into a strict sequence of actions for the agent to execute.
    
    Goal: {goal}
    Environment: {caption}
    
    Provide the steps in a structured format, such as:
    1. Move forward 0.1 units.
    2. Turn right 2.5 degrees.
    3. Move forward 0.1 units.
    4. Pick up the object with the left hand.

    The agent is able to do the following actions
    * move_forward - Moves the agent 0.1 units forward.
    * move_backward - Transforms the agent 0.1 units backward.
    * move_left - Moves the agent 0.1 units to the left.
    * move_right - Moves the agent 0.1 units to the right.
    * pan_left - Pans the camera's agent to look 2.5 degrees to the left.
    * pan_right - Pans the camera's agent to look 2.5 degrees to the right.
    * pan_up - Pans the camera's agent to look 2.5 degrees upwards. 
    * pan_down - Pans the camera's agent to look 2.5 degrees downwards.
    * extend_left_hand_forward - Extends the agent's left hand forward by 0.025 units.
    * extend_right_hand_forward - Extends the agent's right hand forward by 0.025 units.
    * pull_left_hand_backward - Pulls the agent's left hand backward by 0.025 units.
    * pull_right_hand_backward - Pulls the agent's right hand backward by 0.025 units.
    * raise_left_hand - Raises the agent's left hand by 0.025 units.
    * raise_right_hand - Raises the agent's right hand by 0.025 units.
    * lower_left_hand - Lowers the agent's left hand by 0.025 units.
    * lower_right_hand - Lowers the agent's right hand by 0.025 units.
    * toggle_left_grip - Toggles the grip of the left hand.
    * toggle_right_grip - Toggles the grip of the right hand.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(HERMES_DEVICE)
    with torch.inference_mode():
        output_ids = hermes_model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return {"plan": response}


# Have the agent select which tools.
@app.post("/decide_action")
async def predict(prompt: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read image bytes
        image_bytes = await file.read()

        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Format input as per Hermes's expected structure
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

