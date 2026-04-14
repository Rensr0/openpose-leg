import os
import sys
import uuid
import json
import platform
import subprocess
import re
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.body import Body
from src import util

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'output')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_IMAGE = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

MIME_TYPES = {
    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.bmp': 'image/bmp', '.webp': 'image/webp', '.gif': 'image/gif',
    '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mov': 'video/quicktime',
    '.mkv': 'video/x-matroska', '.webm': 'video/webm',
}

body_estimation = None


def get_body():
    global body_estimation
    if body_estimation is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'body_pose_model.pth')
        body_estimation = Body(model_path)
    return body_estimation


@app.route('/api/detect', methods=['POST'])
def detect_frame():
    """实时摄像头检测：接收 base64 JPEG 帧，返回带骨骼标注的 base64 JPEG"""
    data = request.get_json(silent=True)
    if not data or 'frame' not in data:
        return jsonify({'error': '缺少 frame 字段'}), 400

    try:
        img_bytes = base64.b64decode(data['frame'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': '无法解码图片'}), 400
    except Exception as e:
        return jsonify({'error': f'解码失败: {e}'}), 400

    body = get_body()
    candidate, subset = body(frame)
    canvas = frame.copy()
    canvas = util.draw_bodypose(canvas, candidate, subset)

    quality = data.get('quality', 60)
    frame_b64 = encode_jpeg(canvas, quality=quality)
    persons = len(subset)
    keypoints = int(np.sum(subset[:, :-2] >= 0)) if persons > 0 else 0

    return jsonify({
        'frame': frame_b64,
        'persons': persons,
        'keypoints': keypoints,
    })


def allowed_file(filename, allowed):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed


@app.route('/')
def index():
    return render_template('index.html')


def encode_jpeg(frame, quality=75):
    """将帧编码为 base64 JPEG"""
    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return None
    return base64.b64encode(buf.tobytes()).decode('ascii')


@app.route('/api/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    filename = file.filename
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

    task_id = str(uuid.uuid4())[:8]
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{task_id}_{filename}')
    file.save(upload_path)

    if ext in ALLOWED_IMAGE:
        return process_image_stream(upload_path, filename, task_id)
    elif ext in ALLOWED_VIDEO:
        return process_video_stream(upload_path, filename, task_id)
    else:
        return jsonify({'error': f'不支持的格式: {ext}'}), 400


def process_image_stream(upload_path, filename, task_id):
    body = get_body()
    oriImg = cv2.imread(upload_path)
    if oriImg is None:
        return jsonify({'error': '无法读取图片'}), 400

    candidate, subset = body(oriImg)
    canvas = oriImg.copy()
    canvas = util.draw_bodypose(canvas, candidate, subset)

    base_name = filename.rsplit('.', 1)[0]
    output_name = f'{base_name}_legs.png'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
    cv2.imwrite(output_path, canvas)

    frame_b64 = encode_jpeg(canvas, quality=80)

    def generate():
        yield f"data: {json.dumps({'type': 'progress', 'pct': 50, 'label': '正在检测...'})}\n\n"
        yield f"data: {json.dumps({'type': 'frame', 'frame': frame_b64, 'frame_num': 1, 'total_frames': 1})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'output': output_name, 'persons': len(subset), 'keypoints': int(np.sum(subset[:, :-2] >= 0)), 'media_type': 'image'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


def process_video_stream(upload_path, filename, task_id):
    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频'}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = filename.rsplit('.', 1)[0]
    output_name = f'{base_name}_legs.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None
    body = get_body()

    skip = 2
    # 发送每个已处理帧的预览（不做节流，推理本身就跳帧了）
    preview_every_n = 1

    def generate():
        nonlocal writer
        import time as _time

        frame_num = 0
        processed = 0
        last_canvas = None
        next_progress_pct = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % skip == 0:
                candidate, subset = body(frame)
                canvas = frame.copy()
                canvas = util.draw_bodypose(canvas, candidate, subset)
                last_canvas = canvas
                processed += 1
            else:
                canvas = last_canvas if last_canvas is not None else frame

            if writer is None:
                writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)

            frame_num += 1
            pct = int(frame_num / total_frames * 100) if total_frames > 0 else 0

            # 每个已处理帧都发送预览（skip=2 时实际帧率=fps/2，不会太快）
            if frame_num % skip == 0 or frame_num == total_frames:
                frame_b64 = encode_jpeg(canvas, quality=50)
                yield f"data: {json.dumps({'type': 'progress', 'pct': pct, 'label': f'处理中 {frame_num}/{total_frames} 帧', 'frame_num': frame_num, 'total_frames': total_frames})}\n\n"
                yield f"data: {json.dumps({'type': 'frame', 'frame': frame_b64, 'frame_num': frame_num, 'total_frames': total_frames})}\n\n"
            elif pct >= next_progress_pct:
                next_progress_pct = pct + 5
                yield f"data: {json.dumps({'type': 'progress', 'pct': pct, 'label': f'处理中 {frame_num}/{total_frames} 帧', 'frame_num': frame_num, 'total_frames': total_frames})}\n\n"

        cap.release()
        if writer:
            writer.release()

        yield f"data: {json.dumps({'type': 'done', 'output': output_name, 'total_frames': total_frames, 'processed_frames': processed, 'fps': round(fps, 2), 'resolution': f'{width}x{height}', 'media_type': 'video'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


VIDEO_CACHE = {}


@app.route('/api/video-stream/<filename>')
def video_stream(filename):
    """帧流：逐帧推送 JPEG，支持 ?start=N 跳到指定帧"""
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.isfile(path):
        return jsonify({'error': '文件不存在'}), 404

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频'}), 400

    start = request.args.get('start', 0, type=int)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    boundary = 'f'

    def generate():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if not ret:
                    continue
                data = buf.tobytes()
                yield (b'--' + boundary.encode() + b'\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' +
                       data + b'\r\n')
        finally:
            cap.release()
        yield b'--' + boundary.encode() + b'--\r\n'

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=' + boundary,
                    headers={'Cache-Control': 'no-cache'})


@app.route('/api/video-meta/<filename>')
def video_meta(filename):
    """返回视频元信息：总帧数、fps"""
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.isfile(path):
        return jsonify({'error': '文件不存在'}), 404
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return jsonify({'error': '无法打开视频'}), 400
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return jsonify({'total_frames': total, 'fps': round(fps, 2), 'width': w, 'height': h})


@app.route('/api/video-frame/<filename>/<int:frame_num>')
def video_frame(filename, frame_num):
    """返回视频指定帧的 JPEG"""
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if not os.path.isfile(path):
        return jsonify({'error': '文件不存在'}), 404

    # 复用 cap 对象（避免每帧重新打开）
    if filename not in VIDEO_CACHE or VIDEO_CACHE[filename]['path'] != path:
        if filename in VIDEO_CACHE:
            VIDEO_CACHE[filename]['cap'].release()
        cap = cv2.VideoCapture(path)
        VIDEO_CACHE[filename] = {'cap': cap, 'path': path, 'last': -1}
    else:
        cap = VIDEO_CACHE[filename]['cap']

    info = VIDEO_CACHE[filename]

    # 只有帧号不连续时才 seek（连续播放时直接 read 更快）
    if frame_num != info['last'] + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    info['last'] = frame_num

    if not ret:
        # 到末尾，返回最后一帧
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
        ret, frame = cap.read()
        if not ret:
            return jsonify({'error': '读取帧失败'}), 500

    ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ret:
        return jsonify({'error': '编码失败'}), 500

    return Response(buf.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*'})



@app.route('/api/output/<filename>')
def get_output(filename):
    """支持 HTTP Range 的文件输出（视频 seek / 拖进度条）"""
    folder = os.path.abspath(app.config['OUTPUT_FOLDER'])
    path = os.path.join(folder, filename)

    if not os.path.isfile(path):
        return jsonify({'error': '文件不存在'}), 404

    ext = os.path.splitext(filename)[1].lower()
    mime = MIME_TYPES.get(ext, 'application/octet-stream')
    file_size = os.path.getsize(path)

    # Range 请求处理
    range_header = request.headers.get('Range')
    if range_header:
        m = re.match(r'bytes=(\d+)-(\d*)', range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else file_size - 1
            if start >= file_size:
                return Response(status=416)

            end = min(end, file_size - 1)
            chunk_size = end - start + 1

            def generate():
                with open(path, 'rb') as f:
                    f.seek(start)
                    remaining = chunk_size
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            resp = Response(generate(), status=206, mimetype=mime)
            resp.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
            resp.headers['Accept-Ranges'] = 'bytes'
            resp.headers['Content-Length'] = chunk_size
            return resp

    # 非 Range 请求：普通全文件返回
    def generate():
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    resp = Response(generate(), mimetype=mime)
    resp.headers['Accept-Ranges'] = 'bytes'
    resp.headers['Content-Length'] = file_size
    return resp


@app.route('/api/outputs')
def list_outputs():
    files = []
    for f in os.listdir(app.config['OUTPUT_FOLDER']):
        if f == '.gitkeep':
            continue
        path = os.path.join(app.config['OUTPUT_FOLDER'], f)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        files.append({
            'name': f,
            'size': round(size / 1024 / 1024, 2),
            'ext': f.rsplit('.', 1)[1].lower() if '.' in f else '',
        })
    files.sort(key=lambda x: x['name'])
    return jsonify({'files': files})


@app.route('/api/download/<filename>')
def download_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


@app.route('/api/delete/<filename>', methods=['DELETE'])
def delete_output(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'ok': True})
    return jsonify({'error': '文件不存在'}), 404


@app.route('/api/open-folder', methods=['POST'])
def open_folder():
    """打开输出文件夹（仅本地访问有效）"""
    folder = os.path.abspath(app.config['OUTPUT_FOLDER'])
    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(folder)
        elif system == 'Darwin':
            subprocess.Popen(['open', folder])
        else:
            subprocess.Popen(['xdg-open', folder])
        return jsonify({'ok': True, 'path': folder})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
