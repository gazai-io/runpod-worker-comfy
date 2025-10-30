import base64

def save_base64_to_mp4(base64_string, output_file='output.mp4'):
    """
    將 Base64 字串解碼並儲存為 MP4 檔案。
    
    參數:
    - base64_string: str - Base64 編碼的字串。
    - output_file: str - 輸出的 MP4 檔案路徑（預設為 'output.mp4'）。
    
    返回:
    - None，若成功則無例外；否則拋出錯誤。
    """
    try:
        # 解碼 Base64 字串為二進位資料
        decoded_data = base64.b64decode(base64_string)
        
        # 以二進位模式寫入檔案
        with open(output_file, 'wb') as file:
            file.write(decoded_data)
        
        print(f"檔案已成功儲存至 {output_file}。")
    except base64.binascii.Error as e:
        raise ValueError("無效的 Base64 字串。") from e
    except IOError as e:
        raise IOError(f"檔案寫入失敗：{e}") from e

# 使用範例
# 假設 base64_string 是您的 Base64 字串
base64_string = '您的 Base64 字串在此'  # 請替換為實際字串
save_base64_to_mp4(base64_string, 'example.mp4')