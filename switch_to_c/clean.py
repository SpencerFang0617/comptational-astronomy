# 刪除當前資料夾及其中的子資料夾的所有.o, .exe, .txt檔案
import os
def clean():
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.o') or file.endswith('.exe') or file.endswith('.txt'):
                os.remove(os.path.join(root, file))
                print(f"Deleted: {os.path.join(root, file)}")
if __name__ == "__main__":
    clean()