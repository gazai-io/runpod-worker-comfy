import os
import json


json_path = "./test_resources/workflows/workflow_wan2_2_14B_endtoend.json"

with open(json_path, "r", encoding="utf-8") as f:
    workflow_data = json.load(f)

print("Loaded workflow data:", workflow_data)

png_path = "output_base64_png.json"
video_path = "output_base64_video.json"

with open(png_path, "r", encoding="utf-8") as f:
    png_data = json.load(f)
with open(video_path, "r", encoding="utf-8") as f:
    video_data = json.load(f)

workflow_data["input"]["images"] = []
workflow_data["input"]["images"].append(png_data)
workflow_data["input"]["images"].append(video_data)

# save combined workflow
output_workflow_path = "./test_resources/workflows/workflow_wan2_2_14B_endtoend_with_images.json"
with open(output_workflow_path, "w", encoding="utf-8") as f:
    json.dump(workflow_data, f, indent=4, ensure_ascii=False)