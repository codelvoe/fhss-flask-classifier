import os
import io
import time
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from waitress import serve

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, session, send_from_directory, jsonify
)
from flask_session import Session
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18


# ------------------------------
# Flask 基本配置 - 优化大文件处理
# ------------------------------
app = Flask(__name__)
app.secret_key = "fhss-flask-secret-key"

# 配置Flask-Session使用文件系统存储
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = tempfile.gettempdir() + '/fhss_sessions'
app.config['SESSION_FILE_THRESHOLD'] = 500  # 单个session文件最大500个对象
app.config['SESSION_FILE_MODE'] = 384  # 文件权限
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24小时会话

# 初始化Flask-Session
Session(app)

# 增加文件大小限制和超时设置
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1年缓存

# 临时图片存储目录
TEMP_IMAGE_DIR = Path(tempfile.gettempdir()) / "fhss_images"
TEMP_IMAGE_DIR.mkdir(exist_ok=True)

# 批处理配置
BATCH_SIZE = 10  # 每批处理的图片数量
MAX_IMAGES_PER_REQUEST = 200  # 单次最大图片数量


# ------------------------------
# 设备 & 标签 & 模型配置
# ------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX_TO_LABEL: Dict[int, str] = {
    0: 'AIR2s', 1: 'T0000', 2: 'T0001', 3: 'T0010', 4: 'T0011',
    5: 'T0100', 6: 'T0101', 7: 'T0110', 8: 'T0111', 9: 'T1000',
    10: 'T10000', 11: 'T10001', 12: 'T1001', 13: 'T10010', 14: 'T10011',
    15: 'T1010', 16: 'T10100', 17: 'T10101', 18: 'T1011', 19: 'T10110',
    20: 'T10111', 21: 'T1100', 22: 'T11000', 23: 'T1101', 24: 'T1110',
    25: 'T1111', 26: 'a', 27: 'b', 28: 'c', 29: 'd', 30: 'jumper', 31: 'panda'
}

MODEL_CONFIGS = {
    'ResNet18': {
        'path': r"model\resnet18-rfa-bd.pth",
        'description': 'ResNet18 卷积神经网络',
        'preprocess': 'simple'
    },
    'ResTransformer': {
        'path': r"model\ResTR-rfa-db-20.pth",
        'description': 'CNN+Transformer 混合架构',
        'preprocess': 'imagenet'
    }
}


# ------------------------------
# 模型定义：CNN + Transformer
# ------------------------------
class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int = 8,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 ffn_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        modules = list(backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.feat_dim = backbone.fc.in_features
        self.seq_len = seq_len
        self.proj = nn.Linear(self.feat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        chunk = w // self.seq_len
        feats: List[torch.Tensor] = []
        for i in range(self.seq_len):
            patch = x[:, :, :, i * chunk:(i + 1) * chunk]
            f = self.cnn(patch).view(b, self.feat_dim)
            feats.append(f)
        seq = torch.stack(feats, dim=1)
        x_proj = self.proj(seq)
        cls = self.cls_token.expand(b, -1, -1)
        x_cat = torch.cat([cls, x_proj], dim=1) + self.pos_embed
        x_enc = self.transformer(x_cat)
        cls_out = x_enc[:, 0, :]
        return self.fc(cls_out)


# ------------------------------
# 全局模型缓存（避免重复加载）
# ------------------------------
_MODEL_CACHE: Dict[str, nn.Module] = {}


def get_transform(model_name: str) -> transforms.Compose:
    cfg = MODEL_CONFIGS[model_name]
    if cfg['preprocess'] == 'imagenet':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def load_model(model_name: str) -> Optional[nn.Module]:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    cfg = MODEL_CONFIGS[model_name]
    model_path = cfg['path']
    if not os.path.exists(model_path):
        return None

    if model_name == 'ResNet18':
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(IDX_TO_LABEL))
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        model = CNNTransformerClassifier(num_classes=len(IDX_TO_LABEL))
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    _MODEL_CACHE[model_name] = model
    return model


def predict_with_unknown_detection(model: nn.Module, image_tensor: torch.Tensor, threshold: float) -> Dict:
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

    max_prob = float(np.max(probabilities))
    predicted_idx = int(np.argmax(probabilities))

    if max_prob < threshold:
        return {
            'final_prediction': 'Unknown',
            'final_idx': -1,
            'is_unknown': True,
            'max_confidence': max_prob,
            'raw_predicted_idx': predicted_idx,
            'raw_predicted_label': IDX_TO_LABEL[predicted_idx],
            'all_probabilities': probabilities.tolist(),
            'threshold_gap': float(threshold - max_prob)
        }
    return {
        'final_prediction': IDX_TO_LABEL[predicted_idx],
        'final_idx': predicted_idx,
        'is_unknown': False,
        'max_confidence': max_prob,
        'raw_predicted_idx': predicted_idx,
        'raw_predicted_label': IDX_TO_LABEL[predicted_idx],
        'all_probabilities': probabilities.tolist(),
        'threshold_gap': 0.0
    }


def _extract_true_label_from_filename(filename: str) -> str:
    return filename.split('_')[0] if '_' in filename else '未知'


def _save_image_temp(img: Image.Image, filename: str) -> str:
    """将图片保存到临时目录，返回临时文件名"""
    # 生成唯一文件名
    timestamp = int(time.time() * 1000)
    safe_name = secure_filename(filename)
    temp_filename = f"{timestamp}_{safe_name}"
    temp_path = TEMP_IMAGE_DIR / temp_filename
    
    # 保存图片 - 减少图片质量以节省存储空间和内存
    img.save(temp_path, format='JPEG', quality=70, optimize=True)
    
    return temp_filename


def process_single_image(file, model, tfm, threshold: float, idx: int) -> Dict:
    """处理单张图片"""
    filename = secure_filename(file.filename)
    try:
        img = Image.open(file.stream).convert('RGB')
        
        # 限制图片尺寸以减少内存使用
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        input_tensor = tfm(img).unsqueeze(0).to(DEVICE)

        start = time.time()
        pred = predict_with_unknown_detection(model, input_tensor, threshold)
        infer_ms = (time.time() - start) * 1000.0

        predicted_label = pred['final_prediction']
        true_label = _extract_true_label_from_filename(filename)

        if pred['is_unknown']:
            is_correct = None
            status_icon = '❓'
        else:
            is_correct = bool(predicted_label == true_label)
            status_icon = '✅' if is_correct else '❌'

        # 保存图片到临时目录
        temp_filename = _save_image_temp(img, filename)

        return {
            'index': idx + 1,
            'filename': filename,
            'temp_filename': temp_filename,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': float(pred['max_confidence']),
            'inference_time': float(infer_ms),
            'is_correct': is_correct,
            'is_unknown': bool(pred['is_unknown']),
            'status_icon': status_icon,
            'prediction_details': pred,
        }
    except Exception as e:
        print(f"处理图片 {filename} 时出错: {str(e)}")
        return {
            'index': idx + 1,
            'filename': filename,
            'temp_filename': '',
            'true_label': _extract_true_label_from_filename(filename),
            'predicted_label': '错误',
            'confidence': 0.0,
            'inference_time': 0.0,
            'is_correct': False,
            'is_unknown': False,
            'status_icon': '❌',
            'prediction_details': {
                'error': str(e)
            },
        }


def process_uploaded_files_batch(files: List, model_name: str, threshold: float) -> List[Dict]:
    """批量处理上传的文件，分批进行以减少内存压力"""
    model = load_model(model_name)
    if model is None:
        raise FileNotFoundError(f"模型文件不存在: {MODEL_CONFIGS[model_name]['path']}")

    tfm = get_transform(model_name)
    results: List[Dict] = []

    # 限制最大图片数量
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"单次最多只能处理 {MAX_IMAGES_PER_REQUEST} 张图片，当前上传了 {len(files)} 张")

    print(f"开始处理 {len(files)} 张图片...")

    # 分批处理
    for batch_start in range(0, len(files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(files))
        batch_files = files[batch_start:batch_end]
        
        print(f"处理批次 {batch_start//BATCH_SIZE + 1}: {batch_start+1}-{batch_end}")
        
        for local_idx, file in enumerate(batch_files):
            global_idx = batch_start + local_idx
            result = process_single_image(file, model, tfm, threshold, global_idx)
            results.append(result)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 批次间稍作停顿，避免内存压力过大
        if batch_end < len(files):
            time.sleep(0.1)

    print(f"处理完成，共 {len(results)} 个结果")
    return results


def recompute_with_threshold(results: List[Dict], threshold: float) -> List[Dict]:
    updated: List[Dict] = []
    for r in results:
        new_r = dict(r)
        conf = float(r.get('confidence', 0.0))
        details = r.get('prediction_details', {})
        if conf < threshold:
            new_r['predicted_label'] = 'Unknown'
            new_r['is_unknown'] = True
            new_r['status_icon'] = '❓'
            new_r['is_correct'] = None
        else:
            raw_label = details.get('raw_predicted_label', r.get('predicted_label', ''))
            new_r['predicted_label'] = raw_label
            new_r['is_unknown'] = False
            is_correct = bool(raw_label == r.get('true_label'))
            new_r['is_correct'] = is_correct
            new_r['status_icon'] = '✅' if is_correct else '❌'
        updated.append(new_r)
    return updated


def compute_stats(results: List[Dict], threshold: float) -> Dict:
    if not results:
        return {
            'total': 0, 'known': 0, 'unknown': 0, 'correct': 0, 'wrong': 0,
            'accuracy': 0.0, 'unknown_rate': 0.0, 'total_time': 0.0, 'avg_time': 0.0,
        }

    total = len(results)
    unknown = sum(1 for r in results if r['is_unknown'])
    known = total - unknown
    correct = sum(1 for r in results if r['is_correct'] is True)
    wrong = known - correct
    accuracy = (correct / known) * 100.0 if known > 0 else 0.0
    total_time = float(sum(r.get('inference_time', 0.0) for r in results))
    avg_time = total_time / total if total > 0 else 0.0

    return {
        'total': total,
        'known': known,
        'unknown': unknown,
        'correct': correct,
        'wrong': wrong,
        'accuracy': accuracy,
        'unknown_rate': (unknown / total) * 100.0 if total > 0 else 0.0,
        'total_time': total_time,
        'avg_time': avg_time,
    }


def compute_histogram_data(results: List[Dict]) -> Tuple[List[float], List[bool]]:
    confidences = [float(r.get('confidence', 0.0)) for r in results]
    unknown_flags = [bool(r.get('is_unknown', False)) for r in results]
    return confidences, unknown_flags


def compute_threshold_sensitivity(results: List[Dict]) -> Tuple[List[float], List[int]]:
    thresholds = list(np.round(np.arange(0.1, 0.95, 0.05), 2))
    counts: List[int] = []
    confs = [float(r.get('confidence', 0.0)) for r in results]
    for th in thresholds:
        counts.append(int(sum(c < th for c in confs)))
    return thresholds, counts


def compute_confusion_matrix(results: List[Dict]) -> Dict:
    known_results = [r for r in results if not r.get('is_unknown')]
    if not known_results:
        return {"labels": [], "matrix": []}

    labels = sorted(list(set([r['true_label'] for r in known_results] + [r['predicted_label'] for r in known_results])))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for r in known_results:
        ti = label_to_idx[r['true_label']]
        pi = label_to_idx[r['predicted_label']]
        mat[ti, pi] += 1
    return {"labels": labels, "matrix": mat.tolist()}


def get_session_id() -> str:
    if 'sid' not in session:
        session['sid'] = f"s_{int(time.time()*1000)}_{os.getpid()}"
    return session['sid']


def cleanup_temp_images():
    """清理临时图片文件"""
    try:
        if TEMP_IMAGE_DIR.exists():
            # 清理超过24小时的文件
            current_time = time.time()
            for file_path in TEMP_IMAGE_DIR.iterdir():
                if file_path.is_file():
                    if current_time - file_path.stat().st_mtime > 86400:  # 24小时
                        file_path.unlink()
                        
            # 如果目录为空，重新创建
            if not any(TEMP_IMAGE_DIR.iterdir()):
                shutil.rmtree(TEMP_IMAGE_DIR)
                TEMP_IMAGE_DIR.mkdir(exist_ok=True)
    except Exception as e:
        print(f"清理临时文件失败: {e}")


def cleanup_expired_sessions():
    """清理过期的session文件"""
    try:
        session_dir = Path(app.config['SESSION_FILE_DIR'])
        if session_dir.exists():
            current_time = time.time()
            for file_path in session_dir.iterdir():
                if file_path.is_file() and file_path.suffix == '.session':
                    # 检查文件修改时间，超过24小时则删除
                    if current_time - file_path.stat().st_mtime > 86400:
                        file_path.unlink()
                        print(f"已清理过期session文件: {file_path.name}")
    except Exception as e:
        print(f"清理session文件失败: {e}")


# ------------------------------
# 路由
# ------------------------------
@app.route('/', methods=['GET'])
def index():
    sid = get_session_id()
    # 每个会话状态 - 只存储必要的数据，不存储图片
    current_model_name: Optional[str] = session.get('current_model_name')
    confidence_threshold: float = float(session.get('confidence_threshold', 0.8))
    results: List[Dict] = session.get('prediction_results', [])

    stats = compute_stats(results, confidence_threshold)
    confidences, unknown_flags = compute_histogram_data(results)
    th_x, th_y = compute_threshold_sensitivity(results) if results else ([], [])
    cm = compute_confusion_matrix(results)

    # 单样本分析所需默认项
    selected_filename = request.args.get('inspect')
    selected_result = None
    if results:
        if selected_filename:
            selected_result = next((r for r in results if r['filename'] == selected_filename), results[0])
        else:
            selected_result = results[0]

    return render_template(
        'index.html',
        device=str(DEVICE),
        model_configs=MODEL_CONFIGS,
        model_name=current_model_name,
        model_loaded=(current_model_name is not None and load_model(current_model_name) is not None),
        idx2label=IDX_TO_LABEL,
        threshold=confidence_threshold,
        results=results,
        stats=stats,
        confidences=confidences,
        unknown_flags=unknown_flags,
        th_x=th_x,
        th_y=th_y,
        cm=cm,
        selected_result=selected_result,
        max_images=MAX_IMAGES_PER_REQUEST,
    )


@app.route('/image/<filename>')
def serve_image(filename):
    """提供临时图片文件"""
    return send_from_directory(TEMP_IMAGE_DIR, filename)


@app.route('/load_model', methods=['POST'])
def route_load_model():
    model_name = request.form.get('model_name')
    if not model_name or model_name not in MODEL_CONFIGS:
        flash('请选择正确的模型名称', 'error')
        return redirect(url_for('index'))

    model = load_model(model_name)
    if model is None:
        flash(f"模型文件不存在: {MODEL_CONFIGS[model_name]['path']}", 'error')
        return redirect(url_for('index'))

    session['current_model_name'] = model_name
    flash(f"{model_name} 模型加载成功", 'success')
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def route_predict():
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('请先选择图片文件', 'warning')
        return redirect(url_for('index'))

    model_name = session.get('current_model_name')
    if not model_name:
        flash('请先加载模型', 'warning')
        return redirect(url_for('index'))

    # 检查文件数量限制
    if len(files) > MAX_IMAGES_PER_REQUEST:
        flash(f'单次最多只能处理 {MAX_IMAGES_PER_REQUEST} 张图片，当前上传了 {len(files)} 张', 'error')
        return redirect(url_for('index'))

    try:
        threshold = float(request.form.get('threshold', session.get('confidence_threshold', 0.8)))
        
        # 使用优化的批处理函数
        results = process_uploaded_files_batch(files, model_name, threshold)
        
        session['prediction_results'] = results
        session['confidence_threshold'] = threshold
        flash(f'批量预测完成，共处理 {len(results)} 张图片', 'success')
    except ValueError as ve:
        flash(str(ve), 'error')
    except Exception as e:
        flash(f'预测失败: {str(e)}', 'error')

    return redirect(url_for('index'))


@app.route('/clear', methods=['POST'])
def route_clear():
    session['prediction_results'] = []
    cleanup_temp_images()  # 清理临时图片
    flash('结果已清空', 'success')
    return redirect(url_for('index'))


@app.route('/recalc', methods=['POST'])
def route_recalc():
    threshold = float(request.form.get('threshold', session.get('confidence_threshold', 0.8)))
    results = session.get('prediction_results', [])
    if results:
        session['prediction_results'] = recompute_with_threshold(results, threshold)
        session['confidence_threshold'] = threshold
        flash('已使用新阈值重新计算', 'success')
    else:
        session['confidence_threshold'] = threshold
        flash('阈值已更新', 'success')
    return redirect(url_for('index'))


@app.route('/export', methods=['GET'])
def route_export():
    results = session.get('prediction_results', [])
    if not results:
        flash('没有可导出的结果', 'warning')
        return redirect(url_for('index'))

    export_rows = []
    for r in results:
        export_rows.append({
            '序号': r['index'],
            '文件名': r['filename'],
            '真实标签': r['true_label'],
            '预测结果': r['predicted_label'],
            '置信度': r['confidence'],
            '耗时(ms)': r['inference_time'],
            '是否未知': r['is_unknown'],
            '预测正确': r['is_correct'],
        })

    df = pd.DataFrame(export_rows)
    csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    return send_file(
        io.BytesIO(csv_bytes),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"fhss_prediction_results_{int(time.time())}.csv"
    )


# 添加健康检查和状态监控API
@app.route('/api/status')
def api_status():
    """系统状态API"""
    try:
        # 检查临时目录
        temp_files_count = len(list(TEMP_IMAGE_DIR.glob("*")))
        
        # 检查GPU内存（如果可用）
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            }
        
        return jsonify({
            'status': 'ok',
            'device': str(DEVICE),
            'temp_files_count': temp_files_count,
            'gpu_memory': gpu_memory,
            'max_images_per_request': MAX_IMAGES_PER_REQUEST,
            'batch_size': BATCH_SIZE
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    # 在启动时清理临时文件
    cleanup_temp_images()
    
    # 清理过期的session文件
    cleanup_expired_sessions()
    
    # 在开发模式下运行: python app.py
    # 生产部署推荐: gunicorn 或 waitress 等
    # 增加线程数以支持更多并发请求
    #app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    print("Starting waitress server...")  # 调试输出，确认是否启动了 waitress
    serve(app, host='0.0.0.0', port=8000)
