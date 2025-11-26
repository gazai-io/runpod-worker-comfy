import os
import yaml

# 目標搜索資料夾（可依環境調整）
TARGET_DIR = '/runpod-volume/huggingface-cache/'

# YAML 檔案路徑（假設位於 /comfyui 目錄下）
YAML_PATH = '/comfyui/extra_model_paths.yaml'

# allow ext
TARGET_EXTENSIONS = [".safetensors", ".ckpt", ".pth", ".bin"]

TARGET_model_names = ["checkpoints","clip","clip_vision","configs","controlnet","embeddings","loras","upscale_models","vae","unet","ipadapter","conditioning","text_encoders","instantid","insightface","ultralytics","ultralytics_bbox","ultralytics_segm","llama","sams"]

def search_and_update_yaml():
    # 載入原有 YAML 內容（若檔案不存在，初始化為正確結構）
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    config["search_model"] = {}
    config["search_model"]["base_path"] = os.path.abspath(TARGET_DIR)
    # 搜索目標資料夾下的子資料夾
    paths = []
    if os.path.exists(TARGET_DIR):
        for root, dirs, files in os.walk(TARGET_DIR):
            for file in files:
                if any(file.endswith(ext) for ext in TARGET_EXTENSIONS):
                    base_dir_rel_path = os.path.relpath(TARGET_DIR, root)
                    paths.append(base_dir_rel_path)
                    break
    else:
        print(f"目標資料夾 {TARGET_DIR} 不存在。")

    # 更新 YAML 結構
    for model_name in TARGET_model_names:
        if(len(paths) == 1):
            config["search_model"][model_name] = paths[0]
        elif(len(paths) > 1):
            config["search_model"][model_name] = '|\n' + '\n'.join(f"    {p}" for p in sorted(paths))

    # 寫回 YAML 檔案
    with open(YAML_PATH, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)
        print(f"已更新 YAML 檔案：{YAML_PATH}")
    # 輸出最終 YAML 內容
    print("\n最終 YAML 內容：\n")
    with open(YAML_PATH, 'r') as f:
        final_content = f.read()
        print(final_content)

if __name__ == "__main__":
    search_and_update_yaml()

