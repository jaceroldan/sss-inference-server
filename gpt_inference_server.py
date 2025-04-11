import base64
from fastapi import FastAPI, File, UploadFile, Form
from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = FastAPI()

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
    system_prompt = """
    You are an AI planning agent in a virtual grocery environment.
    Break down the user's goal into a strict, numbered sequence of physical actions for the agent.
    Use action phrases like 'Move forward', 'Turn right', 'Pick up item', etc.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Goal: {goal}\nEnvironment: {caption}"}
        ],
        max_tokens=512
    )
    return {"plan": response.choices[0].message.content}


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
