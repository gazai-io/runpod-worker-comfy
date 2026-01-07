import os
import yaml

# 目標搜索資料夾（可依環境調整）
TARGET_DIR = '/runpod-volume/huggingface-cache/'
# TARGET_DIR = r"C:\Users\user\Documents\GitHub\ComfyUI\models" # for testing

# YAML 檔案路徑（假設位於 /comfyui 目錄下）
YAML_PATH = '/comfyui/extra_model_paths.yaml'
# YAML_PATH = './src/extra_model_paths.yaml' # for testing

# allow ext
TARGET_EXTENSIONS = [".safetensors", ".ckpt", ".pth", ".bin"]

TARGET_model_names = [
    "checkpoints","clip","clip_vision","configs","controlnet","embeddings",
    "loras","upscale_models","vae","unet","ipadapter","conditioning","text_encoders",
    "instantid","insightface","ultralytics","ultralytics_bbox","ultralytics_segm","llama","sams"
]

# 載入現有 YAML 配置
with open(YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)
# 根據現有 YAML 內容，追加TARGET_model_names
for set_name in config:
    for model_name in config.get(set_name, {}):
        if model_name not in TARGET_model_names and model_name != "base_path":
            TARGET_model_names.append(model_name)
            print(f"Added model_name from existing yaml: {model_name}")


def search_and_update_yaml():
    # 搜索目標資料夾下的子資料夾
    paths = []
    if os.path.exists(TARGET_DIR):
        for root, dirs, files in os.walk(TARGET_DIR):
            for file in files:
                if any(file.endswith(ext) for ext in TARGET_EXTENSIONS):
                    base_dir_rel_path = os.path.relpath(root, TARGET_DIR)
                    paths.append(base_dir_rel_path)
                    break
    else:
        print(f"目標資料夾 {TARGET_DIR} 不存在。")
    
    return paths

def get_block_yaml(block, block_name):
    yaml_lines = [f"{block_name}:"]
    base_path = block["base_path"]
    yaml_lines.append("  base_path: " + base_path)
    for model_name in TARGET_model_names:
        if model_name not in block:
            continue
        value = block[model_name]
        if "\n" in value:  # 多行路徑，使用區塊標記
            yaml_lines.append(f"  {model_name}: |")
            # 分割多行並添加縮排
            indented_paths = [f"    {line.rstrip()}" for line in value.split("\n") if line.strip()]
            yaml_lines.extend(indented_paths)
        else:  # 單一路徑
            yaml_lines.append(f"  {model_name}: {value}")
    return yaml_lines

def output_yaml(paths, yaml_path=YAML_PATH):
    # 載入原有 YAML 內容（若檔案不存在，初始化為正確結構）
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 更新 search_model 區塊
    search_model = {}  # 初始化或提取原有 search_model
    if "search_model" in config:
        search_model = config["search_model"]
    search_model["base_path"] = os.path.abspath(TARGET_DIR)
    
    # 更新 YAML 結構
    for model_name in TARGET_model_names:
        if len(paths) == 1:
            search_model[model_name] = paths[0]
        elif len(paths) > 1:
            # 調整為 YAML 多行格式，確保正確縮排（原程式碼可能有誤）
            search_model[model_name] = "\n    " + "\n    ".join(p for p in sorted(paths))
    
    # 建立新有序配置，將 search_model 置於首位
    new_config = {}
    new_config["search_model"] = search_model
    for key, value in config.items():
        if key != "search_model":
            new_config[key] = value
    
    # 手動建構 YAML 格式字串（依新順序處理區塊）
    block_yaml_lines = []
    for block_name in new_config:
        block_yaml_lines.extend(get_block_yaml(new_config[block_name], block_name))
    yaml_content = "\n".join(block_yaml_lines) + "\n"
    
    # 將 YAML 格式內容寫入 YAML 檔案
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"已更新 YAML 檔案（YAML 格式）：{yaml_path}")
    
    # 輸出最終 YAML 內容
    print("\n最終 YAML 內容：\n")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        final_content = f.read()
        print(final_content)

if __name__ == "__main__":
    paths = search_and_update_yaml()
    output_yaml(paths)
    # test_paths_3 = ["models/subdir2", "models/subdir1", "cache/dir3"]
    # output_yaml(test_paths_3, yaml_path="./src/extra_model_paths.yaml")
    
