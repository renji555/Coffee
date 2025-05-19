from flask import Flask, request, jsonify, Response
from flask_cors import CORS # 需要安装 flask-cors

app = Flask(__name__)
CORS(app)  # 启用 CORS，允许跨域请求

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        # 验证请求格式
        if not request.is_json:
            return jsonify({"error": "Request must be JSON", "code": 400}), 400
            
        # 获取 JSON 数据
        data = request.get_json()
        
        # 验证是否包含 image 字段
        if 'image' not in data:
            return jsonify({"error": "Missing 'image' in request", "code": 400}), 400
            
        base64_image = data['image']
        
        # TODO: 处理 base64 图像数据
        # 例如：解码图像、进行检测等操作
        # 这里仅作为示例返回成功信息
        
        result = {
            "status": "success",
            "detection": "示例检测结果",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        # 错误处理
        return jsonify({
            "error": str(e),
            "code": 500,
            "status": "error"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)