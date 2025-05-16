import os
import glob

def rename_png_files(folder_path):
    png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    for i, old_path in enumerate(png_files, start=1):
        new_name = f"coffee{str(i).zfill(4)}.png"
        new_path = os.path.join(folder_path, new_name)
        
        os.rename(old_path, new_path)
        print(f"已重命名: {os.path.basename(old_path)} → {new_name}")

if __name__ == "__main__":
    folder_path = "./photo of coffee beans/many beans" 
    rename_png_files(folder_path)