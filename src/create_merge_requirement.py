import os
from pathlib import Path
def add_dependencies(dependencies, req_file):
    """
    將requirements.txt檔案中的依賴項加入集合
    """
    with open(req_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('#')[0].strip()
        if line: # 去除註解和空行
            dependencies.add(line)
            print(f"add_dependencies req_file: {req_file}, added dependency: {line}")
    return dependencies

def merge_requirements():
    # 儲存所有唯一依賴項
    dependencies = set()
    # 轉換為Path物件
    directory = Path("/comfyui/custom_nodes/")
    
    # 遍歷所有requirements.txt檔案
    for dirname in os.listdir(directory):
        dirpath = os.path.join(directory, dirname)
        req_file = os.path.join(dirpath, "requirements.txt")
        if not os.path.isfile(req_file):
            print(f"檔案 {req_file} 不存在，跳過...")
            continue

        dependencies = add_dependencies(dependencies, req_file)

    original_file = "/requirements.txt"
    dependencies = add_dependencies(dependencies, original_file)

    # comfyui主目錄下的requirements.txt
    comfyui_req_file = "/comfyui/requirements.txt"
    dependencies = add_dependencies(dependencies, comfyui_req_file)


    # 將結果寫入新檔案
    output_file="merged_requirements.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 按字母順序排序並寫入
            for dep in sorted(dependencies):
                f.write(f"{dep}\n")
        # print(f"已成功創建合併檔案: {output_file}")
    except Exception as e:
        print(f"寫入檔案 {output_file} 時發生錯誤: {e}")

if __name__ == "__main__":
    merge_requirements()