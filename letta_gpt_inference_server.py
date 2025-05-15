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
async def caption_image(prompt: str = Form(...), image_url: str = Form(...)):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt}\n"
                            "Please describe the scene clearly and concisely, "
                            "highlighting object names, locations, labels, and visible affordances."
                        )
                    },{
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            }
        ],
    )
    return {"response": response.choices[0].message.content}


@app.post("/plan_actions")
async def plan_actions(
    goal: str = Form(...),
    caption: str = Form(...),
    centered: str = Form(...),
    # distance: str = Form(...),
    previous_plans: str = Form(...)
):
    # headers = {
    #     "X-BARE-PASSWORD": f"password {LETTA_RAILWAY_PASSWORD}",
    #     "Content-Type": "application/json",
    # }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a planning assistant for a virtual agent in a grocery environment. You must return a step-by-step plan in simple, numbered steps using only primitive actions."
            },
            {
                "role": "user",
                "content": (
                    f"Environment caption: {caption}\n\n"
                    f"Goal: {goal}\n\n"
                    f"Camera centered on object? {centered}\n\n"
                    # f"Distance to object: {distance}\n\n"
                    f"Previous plans (for context):\n{previous_plans}\n\n"
                    "What are the next precise steps the agent should take to achieve the goal?"
                )
            }
        ]
    )

    # timeout = httpx.Timeout(30.0)  # or even higher if needed
    # async with httpx.AsyncClient(timeout=timeout) as client:
    #     letta_response = await client.post(LETTA_API_URL, headers=headers, json=payload)
    #     letta_data = letta_response.json()

    # You might want to format this more strictly into a list
    # print(letta_data)
    # print(letta_data["messages"])
    # print(len(letta_data["messages"]))
    return {"plan": response.choices[0].message.content}


# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "grab_and_read_item",
            "description": "Extends, grasps, and inspects an object directly in front of the agent using the specified hand. Returns OCR-extracted details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hand": {
                        "type": "string",
                        "enum": ["left"],
                        "description": "The hand to use for grasping the object."
                    }
                },
                "required": ["hand"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "move_forward",
    #     "description": "Moves the agent forward by (units × 0.1) units.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "units": {
    #                 "type": "integer",
    #                 "description": "Number of times to move 0.1 units forward."
    #             }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "move_backward",
    #     "description": "Moves the agent backward by (units × 0.1) units.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to move 0.1 units backward."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "move_left",
    #     "description": "Moves the agent to the left by (units × 0.1) units.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to move 0.1 units to the left."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "move_right",
    #     "description": "Moves the agent to the right by (units × 0.1) units.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to move 0.1 units to the right."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "pan_left",
    #     "description": "Pans the agent's camera to the left by (units × 2.5) degrees.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to pan 2.5 degrees left."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "pan_right",
    #     "description": "Pans the agent's camera to the right by (units × 2.5) degrees.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to pan 2.5 degrees right."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "pan_up",
    #     "description": "Pans the agent's camera upward by (units × 2.5) degrees.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to pan 2.5 degrees upward."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #     "name": "pan_down",
    #     "description": "Pans the agent's camera downward by (units × 2.5) degrees.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #         "units": {
    #             "type": "integer",
    #             "description": "Number of times to pan 2.5 degrees downward."
    #         }
    #         },
    #         "required": ["units"],
    #         "additionalProperties": False
    #     },
    #     "strict": True
    #     }
    # },

    # {
    #     "type": "function",
    #     "function": {
    #         "name": "extend_left_hand_forward",
    #         "description": "Extends the agent's left hand forward by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to move left hand by 0.025 units forward."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "extend_right_hand_forward",
    #         "description": "Extends the agent's right hand forward by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to move right hand by 0.025 units forward."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "pull_left_hand_backward",
    #         "description": "Pulls the agent's left hand backward by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to move left hand by 0.025 units backward."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "pull_right_hand_backward",
    #         "description": "Pulls the agent's right hand backward by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to move right hand by 0.025 units backward."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "raise_left_hand",
    #         "description": "Raises the agent's left hand by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to raise left hand by 0.025 units."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "raise_right_hand",
    #         "description": "Raises the agent's right hand by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to raise right hand by 0.025 units."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "lower_left_hand",
    #         "description": "Lowers the agent's left hand by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to lower left hand by 0.025 units."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "lower_right_hand",
    #         "description": "Lowers the agent's right hand by 0.025 units.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "units": {
    #                     "type": "integer",
    #                     "description": "Number of times to lower right hand by 0.025 units."
    #                 }
    #             },
    #             "required": ["units"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "toggle_left_grip",
    #         "description": "Toggles the grip of the left hand.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "required": [],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "toggle_right_grip",
    #         "description": "Toggles the grip of the right hand.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {},
    #             "required": [],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "center_object_on_screen",
    #         "description": "Centers the agent's camera on the specified object using visual feedback from a CLIP-based object detector.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "target_name": {
    #                     "type": "string",
    #                     "description": "The name or description of the object the agent should center in view."
    #                 }
    #             },
    #             "required": ["target_name"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
    
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "rotate_and_read",
    #         "description": "Inspects an object directly in front of the agent using the specified hand by rotating an already-grabbed object in clockwise direction. Returns OCR-extracted details.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "hand": {
    #                     "type": "string",
    #                     "enum": ["left", "right"],
    #                     "description": "The hand to use for grasping the object."
    #                 }
    #             },
    #             "required": ["hand"],
    #             "additionalProperties": False
    #         },
    #         "strict": True
    #     }
    # },
]


@app.post("/decide_action")
async def decide_action(
        plan: str = Form(...),
        previous_actions: str = Form(...),
        centered: str = Form(...),
        # distance: str = Form(...),
        item_grabbed: str = Form(...),
    ):
    print(item_grabbed)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an AI agent that executes one action at a time "
                            "based on a provided plan. You must select exactly one action "
                            "from the available tools below, considering previous actions taken."
                            "If you have grabbed the item, "
                            "rotate and read so you can get information."
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Plan:\n{plan}\n\n"
                            f"Previous actions:\n{previous_actions}\n\n"
                            f"Camera centered on object:\n{centered}\n\n"
                            # f"Distance to object:\n{distance}\n\n"
                            f"Item grabbed:\n{item_grabbed}\n\n"
                            "What is the next best action?"
                        )
                    }
                ]
            }
        ],
        tools=tools,  # assuming tools is your action schema
        tool_choice="required",
        max_tokens=256
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

# from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# model_id = "IDEA-Research/grounding-dino-base"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
from transformers import pipeline
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

# @app.post("/locate-grounding-dino")
# async def locate_grounding_dino(prompt: str = Form(...), file: UploadFile = File(...)):
#     # Read and load image

#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

#     # "A box of cereal"
#     print(prompt)
#     text_labels = [[prompt]]

#     inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs) 

#     # Locate object using Grounding DINO
#     results = processor.post_process_grounded_object_detection(
#         outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]]
#     )

#     print(len(results), results)
#     result = results[0]
#     objects_detected = []
#     for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
#         box = [round(i, 2) for i in box.tolist()]
#         score = round(score.item(), 3)
#         objects_detected.append((box, score, labels))
#         print(f"Detected {labels} with confidence {score} at location {box}")
    
#     return objects_detected
@app.post("/locate-owl-vit")
async def locate_owl_vit(prompt: str = Form(...), file: UploadFile = File(...)):
    # Read and load image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    predictions = detector(
        image,
        candidate_labels=[prompt],
    )
    print(predictions)
    return predictions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
