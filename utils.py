import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def locate_object_in_frame(image: Image.Image, prompt: str, grid_size=5):
    W, H = image.size
    patch_w, patch_h = W // grid_size, H // grid_size

    text_input = clip.tokenize([prompt]).to(device)
    text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    best_score = -1
    best_patch_center = None

    for i in range(grid_size):
        for j in range(grid_size):
            # Crop patch
            left = i * patch_w
            upper = j * patch_h
            right = left + patch_w
            lower = upper + patch_h
            patch = image.crop((left, upper, right, lower))

            # Preprocess and encode
            patch_input = preprocess(patch).unsqueeze(0).to(device)
            patch_features = model.encode_image(patch_input)
            patch_features /= patch_features.norm(dim=-1, keepdim=True)

            # Similarity
            similarity = (patch_features @ text_features.T).item()

            if similarity > best_score:
                best_score = similarity
                center_x = left + patch_w // 2
                center_y = upper + patch_h // 2
                best_patch_center = (center_x, center_y)

    frame_center = (W // 2, H // 2)
    offset_x = best_patch_center[0] - frame_center[0]
    offset_y = best_patch_center[1] - frame_center[1]

    return {
        "best_similarity": best_score,
        "best_patch_center": best_patch_center,
        "frame_center": frame_center,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }
