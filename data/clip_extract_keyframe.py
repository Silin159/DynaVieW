import os
import cv2
from keyframe_extractor import extract_keyframes

video_root = "<your_path_to_the_downloaded_source_videos>"
out_root = "<your_path_to_save_the_output_keyframes>"
for portion in ["Ego4D", "AgiBotWorld", "ShareGPT4Video"]:
    video_dir = os.path.join(video_root, portion)
    processed = 0
    for file in os.listdir(video_dir):
        if file.endswith(".mp4"):
            video_pth = os.path.join(video_dir, file)
            if portion in ["Ego4D", "ShareGPT4Video"]:
                video_id = "keyframes_"+file[:8]
            else:
                video_id = "keyframes_"+file.split(".")[0]
            save_dir = os.path.join(os.path.join(out_root, portion), video_id)
            os.makedirs(save_dir, exist_ok=True)

            keyframes = extract_keyframes(
                video_path=video_pth,
                keyframe_interval=1,
                sharpness_interval=0.1,
                similarity_threshold=0.925,
            )

            for idx, fr in enumerate(keyframes):
                fid = "keyframe_"+f"{idx:02}"+".jpg"
                cv2.imwrite(os.path.join(save_dir, fid), fr)
            
            processed += 1
            print(f"Done {portion} video {processed}")
