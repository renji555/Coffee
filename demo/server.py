import os
import base64
import json
import subprocess
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = '/Users/renhongyi/Desktop'  # 图片保存目录
YOLO_SCRIPT = '/Users/renhongyi/Desktop/yolov5/detect.py'  # YOLO脚本路径
RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')  # 结果保存目录

# 确保结果目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return send_file('frontend/index.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        # 获取请求中的图像数据
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': '未提供图像数据'}), 400
        
        # 解码base64图像
        image_data = base64.b64decode(data['image'])
        
        # 创建临时文件保存图像
        with tempfile.NamedTemporaryFile(
            dir=UPLOAD_FOLDER,
            prefix='detection_',
            suffix='.jpg',
            delete=False
        ) as temp_file:
            temp_file.write(image_data)
            image_path = temp_file.name
        
        # 构建并执行YOLO命令
        command = [
            'python',
            YOLO_SCRIPT,
            '--source', image_path,
            '--project', RESULTS_FOLDER,
            '--name', 'detected'
        ]
        
        app.logger.info(f"执行命令: {' '.join(command)}")
        
        # 执行命令
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        app.logger.info(f"命令输出: {result.stdout}")
        
        # 解析检测结果
        results = parse_yolo_results(image_path)
        
        # 清理临时文件
        # os.unlink(image_path)  # 取消注释此行以在处理后删除临时文件
        
        return jsonify({'results': results})
    
    except subprocess.CalledProcessError as e:
        app.logger.error(f"命令执行失败: {e.stderr}")
        return jsonify({'error': f"模型执行失败: {e.stderr}"}), 500
    except Exception as e:
        app.logger.error(f"处理请求时出错: {str(e)}")
        return jsonify({'error': f"处理请求时出错: {str(e)}"}), 500

def parse_yolo_results(image_path):
    """解析YOLO检测结果"""
    # 获取对应的txt文件路径
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    txt_path = os.path.join(RESULTS_FOLDER, 'detected', 'labels', f'{name_without_ext}.txt')
    
    # 如果没有检测到对象，返回空列表
    if not os.path.exists(txt_path):
        return []
    
    # 读取类别名称
    with open('data/coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # 读取图像尺寸
    from PIL import Image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # 解析检测结果
    results = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_id = int(parts[0])
            confidence = float(parts[4])
            
            # 将YOLO格式(中心x, 中心y, 宽, 高)转换为边界框格式(x, y, 宽, 高)
            center_x = float(parts[1]) * img_width
            center_y = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            width = int(width)
            height = int(height)
            
            # 确保边界框在图像内
            x = max(0, x)
            y = max(0, y)
            width = min(width, img_width - x)
            height = min(height, img_height - y)
            
            results.append({
                'class': class_names[class_id] if class_id < len(class_names) else f'class_{class_id}',
                'confidence': confidence,
                'box': {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                }
            })
    
    return results



import time
import random
def generate_ai_reply(user_message):
    # 这里可以接入实际的AI模型，如GPT、BERT等
    # 为了演示，我们简单地返回一些预设的回复
    
    # 模拟思考时间
    time.sleep(5)
    
    responses = [
        "反萃取过程是通过去除原料中的水分或杂质，提高干燥效率，常用方法包括干燥、水洗和过滤等步骤。"
    ]
    
    # 随机选择一个回复模板
    if user_message == "请用五十个字说明一下什么是反萃取过程":
        template = responses[0]
    elif user_message == "请用三十字概括阿拉比卡咖啡豆的特点":
        template = "阿拉比卡咖啡豆以高海拔、小粒、果香浓郁著称，富含咖啡因和天然风味。"
    else:
        template = "冲泡一杯苦味、偏酸味的瑰夏豆咖，建议使用85℃左右的水，3分钟内冲泡，豆子粒径约0.5-1毫米，以获得自然的风味和口感。"
    
    
    return template

# 聊天API端点
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "消息不能为空"}), 400
        
        # 生成AI回复
        ai_reply = generate_ai_reply(user_message)
        
        return jsonify({"reply": ai_reply})
    
    except Exception as e:
        print(f"Error processing chat request: {str(e)}")
        return jsonify({"error": "处理请求时出错"}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

