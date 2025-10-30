import base64
import gzip
import json

def compress_file_to_base64(input_file, output_file=None):
    """
    將檔案壓縮並轉換為 Base64 字串。

    參數:
    - input_file: str - 輸入檔案的路徑。
    - output_file: str 或 None - 若指定，則將 Base64 字串儲存至該檔案；否則僅返回字串（預設為 None）。

    返回:
    - str - 壓縮後的 Base64 字串。
    """
    try:
        # 以二進位模式讀取檔案
        with open(input_file, 'rb') as file:
            file_content = file.read()
                
        # 將壓縮內容轉為 Base64 字串
        base64_string = base64.b64encode(file_content).decode('utf-8')
        base64_json = {
            "name": input_file,
            "image": base64_string
        }       
        
        # 若指定輸出檔案，則寫入
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(base64_json, file)
            print(f"Base64 字串已儲存至 {output_file}。")
        
        return base64_string
    except FileNotFoundError as e:
        raise FileNotFoundError(f"找不到輸入檔案：{input_file}") from e
    except IOError as e:
        raise IOError(f"檔案處理失敗：{e}") from e

# 使用範例
# 假設 input_file 是您的檔案路徑
input_file = 'test.png'  # 請替換為實際檔案路徑，例如 'example.mp4'
base64_result = compress_file_to_base64(input_file, 'output_base64.json')
print("壓縮後的 Base64 字串：")
print(base64_result)