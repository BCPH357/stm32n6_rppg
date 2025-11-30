"""
rPPG Web Application - Flask Server
- 提供網頁介面
- WebSocket 通訊（接收影格、返回推論結果）
- 即時心率檢測
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from inference import HeartRateDetector

# 創建 Flask 應用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rppg_heart_rate_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 創建心率檢測器（全局單例）
detector = HeartRateDetector()

print("\n" + "="*60)
print("rPPG Web Application Server")
print("="*60)
print(f"Flask app created")
print(f"SocketIO initialized")
print("="*60)


@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')


@app.route('/status')
def status():
    """系統狀態 API"""
    return jsonify({
        'status': 'running',
        'frame_count': detector.frame_count,
        'bvp_buffer_size': len(detector.bvp_buffer),
        'hr_history_size': len(detector.hr_history)
    })


@socketio.on('connect')
def handle_connect():
    """客戶端連接"""
    print(f"[OK] Client connected")
    emit('message', {'text': '已連接到服務器'})


@socketio.on('disconnect')
def handle_disconnect():
    """客戶端斷開"""
    print("[Disconnect] Client disconnected")


@socketio.on('frame')
def handle_frame(data):
    """
    處理影格數據

    Args:
        data: {
            'image': base64 編碼的 JPEG 圖像 (data:image/jpeg;base64,...)
        }

    Returns:
        emit 'result' 事件，包含推論結果
    """
    try:
        # 解碼 base64 影格
        img_str = data['image']

        # 移除 data URL 前綴（如果有）
        if ',' in img_str:
            img_str = img_str.split(',')[1]

        # Base64 解碼
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)

        # 解碼為 OpenCV 影像
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            emit('error', {'message': '無法解碼影格'})
            return

        # 推論
        result = detector.process_frame(frame)

        # 返回結果給前端
        emit('result', {
            'face_bbox': result['face_bbox'],
            'roi_coords': result['roi_coords'],
            'hr': result['hr'],
            'hr_raw': result['hr_raw'],
            'frame_count': result['frame_count'],
            'hr_history': result['hr_history'],
            'status': result['status']
        })

    except Exception as e:
        print(f"[Error] Processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Error processing frame: {str(e)}'})


@socketio.on('reset')
def handle_reset():
    """重置檢測器"""
    try:
        detector.reset()
        emit('message', {'text': 'Detector reset successfully'})
        print("[OK] Detector reset by client")
    except Exception as e:
        print(f"[Error] Resetting detector: {e}")
        emit('error', {'message': f'Reset failed: {str(e)}'})


@socketio.on('ping')
def handle_ping():
    """心跳檢測"""
    emit('pong', {'timestamp': int(time.time() * 1000)})


if __name__ == '__main__':
    import time

    print("\n" + "="*60)
    print("Starting Flask-SocketIO Server")
    print("="*60)
    print("Server URL: http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    # 啟動服務器
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # 避免重複載入模型
    )
