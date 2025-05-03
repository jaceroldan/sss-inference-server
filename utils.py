import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def locate_object_in_frame(image: Image.Image, prompt: str, grid_size=5):
    W, H = image.size
    patch_w, patch_h = W // grid_size, H // grid_size

    text_input = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        patches = []
        bboxes = []  # (left, upper, right, lower)
        for i in range(grid_size):
            for j in range(grid_size):
                left = i * patch_w
                upper = j * patch_h
                right = min(left + patch_w, W)
                lower = min(upper + patch_h, H)
                patch = image.crop((left, upper, right, lower))
                patches.append(preprocess(patch))
                bboxes.append((left, upper, right, lower))

        patch_inputs = torch.stack(patches).to(device)
        patch_features = model.encode_image(patch_inputs)
        patch_features /= patch_features.norm(dim=-1, keepdim=True)

        similarities = (patch_features @ text_features.T).squeeze()
        best_idx = similarities.argmax().item()

    best_score = similarities[best_idx].item()
    best_bbox = bboxes[best_idx]
    best_patch_center = ((best_bbox[0] + best_bbox[2]) // 2, (best_bbox[1] + best_bbox[3]) // 2)
    frame_center = (W // 2, H // 2)

    return {
        "best_similarity": best_score,
        "best_patch_center": best_patch_center,
        "frame_center": frame_center,
        "offset_x": best_patch_center[0] - frame_center[0],
        "offset_y": best_patch_center[1] - frame_center[1],
        "bounding_box": {
            "x_min": best_bbox[0],
            "y_min": best_bbox[1],
            "x_max": best_bbox[2],
            "y_max": best_bbox[3]
        }
    }
