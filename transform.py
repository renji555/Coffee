from PIL import Image
import os

def convert_images_to_png(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                new_file_path = os.path.splitext(file_path)[0] + '.png'
                # 保存为PNG格式
                img.save(new_file_path, 'PNG')
                os.remove(file_path)
            except Exception as e:
                print(f"Error converting {file_path}: {e}")

if __name__ == "__main__":
    folder_path = "your_folder_path"  # 替换为实际的文件夹路径
    convert_images_to_png("./photo of coffee beans/many beans")