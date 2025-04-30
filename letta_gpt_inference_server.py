import base64
import io

import torch
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image
from openai import OpenAI
import httpx
import clip

from utils import locate_object_in_frame

from dotenv import load_dotenv
import os

load_dotenv()

LETTA_AGENT_ID = os.getenv("LETTA_AGENT_ID")
LETTA_API_URL = f"https://lettalettalatest-production-b515.up.railway.app/v1/agents/{LETTA_AGENT_ID}/messages"
LETTA_RAILWAY_PASSWORD = os.getenv("LETTA_RAILWAY_PASSWORD")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


class CLIPRequest(BaseModel):
    prompt: str


@app.post("/caption_image")
async def caption_image(prompt: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
    )
    return {"response": response.choices[0].message.content}


@app.post("/plan_actions")
async def plan_actions(goal: str = Form(...), caption: str = Form(...)):
    headers = {
        "X-BARE-PASSWORD": f"password {LETTA_RAILWAY_PASSWORD}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given this environment: {caption}\nAnd this goal: {goal}\nWhat are the precise steps I should take?"
                    }
                ]
            }
        ]
    }
    timeout = httpx.Timeout(30.0)  # or even higher if needed
    async with httpx.AsyncClient(timeout=timeout) as client:
        letta_response = await client.post(LETTA_API_URL, headers=headers, json=payload)
        letta_data = letta_response.json()

    # You might want to format this more strictly into a list
    print(letta_data)
    print(letta_data["messages"])
    print(len(letta_data["messages"]))
    return {"plan": letta_data["messages"]}


# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "move_forward",
            "description": "Moves the agent 0.1 units forward.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_backward",
            "description": "Moves the agent 0.1 units backward.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_left",
            "description": "Moves the agent 0.1 units to the left.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_right",
            "description": "Moves the agent 0.1 units to the right.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pan_left",
            "description": "Pans the agent's camera 2.5 degrees to the left.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pan_right",
            "description": "Pans the agent's camera 2.5 degrees to the right.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pan_up",
            "description": "Pans the agent's camera 2.5 degrees upward.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pan_down",
            "description": "Pans the agent's camera 2.5 degrees downward.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extend_left_hand_forward",
            "description": "Extends the agent's left hand forward by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extend_right_hand_forward",
            "description": "Extends the agent's right hand forward by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pull_left_hand_backward",
            "description": "Pulls the agent's left hand backward by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pull_right_hand_backward",
            "description": "Pulls the agent's right hand backward by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raise_left_hand",
            "description": "Raises the agent's left hand by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "raise_right_hand",
            "description": "Raises the agent's right hand by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lower_left_hand",
            "description": "Lowers the agent's left hand by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lower_right_hand",
            "description": "Lowers the agent's right hand by 0.025 units.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_left_grip",
            "description": "Toggles the grip of the left hand.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_right_grip",
            "description": "Toggles the grip of the right hand.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


@app.post("/decide_action")
async def decide_action(prompt: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        tools=tools,
        tool_choice="required",
        max_tokens=512
    )

    return {"response": response.choices[0].message.tool_calls if response.choices[0].message.tool_calls else response.choices[0].message.content}


@app.post("/find-object/")
async def find_object(prompt: str, file: UploadFile = File(...)):
    # Read the image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Preprocess
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Encode text
    text_input = clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_input)

    # Encode image
    image_features = model.encode_image(image_input)

    # Compare
    similarity = torch.cosine_similarity(image_features, text_features)
    
    return {"similarity": similarity.item()}


@app.post("/locate-object/")
async def locate_object(prompt: str, file: UploadFile = File(...)):
    # Read and load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Locate object
    result = locate_object_in_frame(image, prompt)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
