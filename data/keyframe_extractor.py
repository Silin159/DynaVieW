import os
import math
import cv2
import numpy as np
import torch
from typing import List
from torch.nn import functional as F
from transformers import CLIPFeatureExtractor, CLIPVisionModel

# Load CLIP model once, move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL_PATH = '<your_path_to_the_downloaded_clip_vit_model>'
_feature_extractor = CLIPFeatureExtractor.from_pretrained(_MODEL_PATH)
_vision_tower = CLIPVisionModel.from_pretrained(_MODEL_PATH).to(device)
_vision_tower.eval()
for p in _vision_tower.parameters(): p.requires_grad = False


def get_resized_wh(width: int, height: int, max_size: int):
    if max(width, height) <= max_size:
        return width, height
    scale = max_size / max(width, height)
    return int(width * scale), int(height * scale)


def check_pure(frame: np.ndarray) -> bool:
    # Fast check for uniform frame
    flat = frame.reshape(-1, frame.shape[-1])
    return np.all(flat == flat[0], axis=1).all()


def get_sharpness(frame: np.ndarray) -> float:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    sharpness = laplacian.var()
    return sharpness


def extract_keyframes(
    video_path: str,
    keyframe_interval: int,
    sharpness_interval: int,
    similarity_threshold: float,
) -> List[np.ndarray]:
    """
    Extract frames at ~keyframe_interval sec where semantic change < window_threshold.
    Returns list of BGR frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, round(fps * keyframe_interval))
    sharp_shift = max(0, round(fps * sharpness_interval))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    start = 0
    for idx in range(total_frames):
        start = idx
        ret, fr = cap.read()
        if ret and not check_pure(fr):
            break
    
    indices = list(range(start, total_frames, frame_step))
    # if indices[-1] != total_frames - 1:
    #     indices.append(total_frames - 1)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx-sharp_shift, start))
        best_sharp = 0
        best_frame = None
        for _ in range(max(idx-sharp_shift, start), min(idx+sharp_shift+1, total_frames)):
            ret, fr = cap.read()
            if not ret or check_pure(fr):
                continue
            sharp = get_sharpness(fr)
            if sharp > best_sharp:
                best_sharp = sharp
                best_frame = fr
        if best_frame is not None:
            frames.append(best_frame)
    
    cap.release()

    if len(frames) == 0:
        return []

    orig_h, orig_w = frames[0].shape[:2]
    new_w, new_h = get_resized_wh(orig_w, orig_h, 1024)
    if (new_w, new_h) != (orig_w, orig_h):
        frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    
    # compute CLIP features in batch
    inputs = _feature_extractor(images=frames, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = _vision_tower(**inputs, output_hidden_states=True).hidden_states[-1][:,0].cpu()
    # feats = F.normalize(out[:,0], dim=1).cpu()

    selected = [0]
    last_idx = 0
    for i in range(1, feats.size(0)):
        sim = F.cosine_similarity(feats[last_idx:last_idx+1], feats[i:i+1], dim=1).item()
        if sim < similarity_threshold:
            selected.append(i)
            last_idx = i

    # ensure last frame if far enough
    # if selected[-1] != len(frames)-1 and (len(frames)-1 - selected[-1]) * frame_step >= frame_step:
    #     selected.append(len(frames)-1)

    return [frames[i] for i in selected]
