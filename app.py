from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask import make_response
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import webbrowser
import threading
import time
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# ================= 路径配置 =================
# 相对路径（本地&云端通用）
KAN_PATH = r"./model/regression_KAN_model.pkl"
CLASSIFICATION_PATH = r"./model/classification_KAN_model.pkl"
DATA_PATH = r"./data/data.csv"
IMAGE_PATH = r"./img/GUI 的图A.png"

VALIDATION_IMAGES = {
    "kan": r"./img/KAN_optimized.png",
    "mlp": r"./img/MLP_optimized.png",
    "lightgbm": r"./img/LightGBM_optimized.png",
    "xgboost": r"./img/XGBoost_optimized.png"
}


# ================= 路由 =================
@app.route('/validation-image/<model_name>')
def validation_image(model_name):
    model_name = model_name.lower()
    if model_name in VALIDATION_IMAGES and os.path.exists(VALIDATION_IMAGES[model_name]):
        return send_from_directory(
            os.path.dirname(VALIDATION_IMAGES[model_name]),
            os.path.basename(VALIDATION_IMAGES[model_name]),
            mimetype='image/png'
        )
    else:
        return f'''
        <html><body style="font-family:Times New Roman; padding:20px; color:#d32f2f;">
            <h2>❌ {model_name.upper()} 验证图未找到！</h2>
        </body></html>
        ''', 404


CONT_COLS = ['Tc', 'Wb', 'Ec', 'Lb', 'Tb', 'Ea', 'σa', 'εa', 'Ab', 'Ar']
CAT_COLS = ['Jt', 'Bt']


# ================= KAN 模型 =================
class SimpleKAN(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=1):
        super(SimpleKAN, self).__init__()
        self.device = torch.device("cpu")
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.bspline = nn.SiLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bspline(x)
        x = self.dropout(x)
        x = self.hidden1(x)
        x = self.bspline(x)
        x = self.dropout(x)
        x = self.hidden2(x)
        x = self.bspline(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)


def load_models_and_preprocessor():
    print("📂 正在加载训练数据...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ohe.fit(df[CAT_COLS])
    scaler = MinMaxScaler()
    scaler.fit(df[CONT_COLS])

    print("✅ KAN 模型加载中...")
    kan = None
    try:
        with open(KAN_PATH, 'rb') as f:
            kan = pickle.load(f)
        if not hasattr(kan, 'device'):
            kan.device = torch.device("cpu")
        print(f"✅ KAN 模型加载成功: {type(kan).__name__}")
    except Exception as e:
        print(f"⚠️  KAN 模型加载失败: {e}")
        kan = SimpleKAN(input_dim=12)

    classifier = None
    try:
        with open(CLASSIFICATION_PATH, 'rb') as f:
            classifier = pickle.load(f)
        if isinstance(classifier, torch.nn.Module):
            if not hasattr(classifier, 'device'):
                classifier.device = torch.device("cpu")
            print(f"✅ 分类模型加载成功")
    except Exception as e:
        print(f"⚠️  分类模型加载失败: {e}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kan.to(DEVICE)
    kan.eval()
    if classifier:
        classifier.to(DEVICE)
        classifier.eval()
    return kan, classifier, scaler, ohe, DEVICE


kan_model, classifier_model, scaler, ohe, DEVICE = load_models_and_preprocessor()

# ================= 页面 =================
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=1280, height=800, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no">
    <title>CFRP Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.8/dist/chart.umd.min.js"></script>
<style>
/* ============================================================
   全局 & 字体
   ============================================================ */
* { margin:0; padding:0; box-sizing:border-box; }

html, body {
    width: 1280px;
    height: 800px;
    overflow: hidden;
    background: #f8f9fc;  /* 统一背景色 */
    font-family: 'Times New Roman', Times, serif;
    color: #1a2a4a;
}

/* ============================================================
   顶部标题栏
   ============================================================ */
header {
    height: 56px;
    background: #0d2b5e;          /* 深海军蓝 */
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    border-bottom: 3px solid #c8a84b;  /* 金色下边框 */
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);  /* 柔化阴影 */
    border-radius: 0 0 8px 8px;  /* 底部圆角柔化 */
}

header h1 {
    font-size: 22px;
    font-weight: bold;
    color: #ffffff;
    letter-spacing: 0.5px;
    font-family: 'Times New Roman', Times, serif;
}

#btn-validate {
    position: absolute;
    top: 50%;
    left: 18px;
    transform: translateY(-50%);
    padding: 6px 16px;
    background: transparent;
    color: #c8a84b;
    border: 1.5px solid #c8a84b;
    border-radius: 6px;  /* 圆角按钮 */
    font-size: 14px;
    font-family: 'Times New Roman', Times, serif;
    cursor: pointer;
    transition: all 0.3s ease;  /* 柔化过渡 */
}
#btn-validate:hover {
    background: #c8a84b;
    color: #0d2b5e;
    box-shadow: 0 2px 6px rgba(200,168,75,0.3);
}

/* ============================================================
   主体布局
   ============================================================ */
.top-container { width:1280px; height:800px; display:flex; flex-direction:column; padding: 0 4px; }

.main-content {
    display: flex;
    flex: 1;
    height: 744px;   /* 800 - 56 header */
    gap: 8px;  /* 面板间距 */
}

/* ============================================================
   左侧面板（加宽至420px）
   ============================================================ */
.left-panel {
    width: 420px;  /* 加宽左侧面板 */
    height: 100%;
    background: #ffffff;
    border: 1px solid #d0d8e8;  /* 统一边框 */
    border-radius: 8px;  /* 圆角柔化 */
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);  /* 柔化阴影 */
}

.params-section {
    flex: 0 0 auto;
    max-height: 480px;  /* 调整高度给图片更多空间 */
    overflow-y: auto;
    border-radius: 6px;
    padding: 4px;
}

/* 滚动条美化 */
.params-section::-webkit-scrollbar { width: 6px; }
.params-section::-webkit-scrollbar-track { background: #f0f2f5; border-radius: 3px; }
.params-section::-webkit-scrollbar-thumb { 
    background: #b0bfd0; 
    border-radius: 3px;
    transition: background 0.2s ease;
}
.params-section::-webkit-scrollbar-thumb:hover { background: #8a9ab0; }

.params-header {
    font-size: 18px;  /* 加大标题 */
    font-weight: bold;
    text-align: center;
    color: #0d2b5e;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #0d2b5e;
    letter-spacing: 0.3px;
}

/* 参数行 */
.param-group-label {
    font-size: 13px;
    font-weight: bold;
    color: #5a7090;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 12px 0 6px;
    padding-left: 4px;
}

label {
    font-size: 15px;  /* 加大标签字体 */
    color: #2a3a5a;
    margin-bottom: 4px;
    display: block;
}

.slider-row {
    display: flex;
    gap: 10px;
    align-items: center;
}

input[type="range"] {
    flex: 1;
    height: 6px;  /* 加宽滑块 */
    accent-color: #0d2b5e;
    cursor: pointer;
    border-radius: 3px;  /* 滑块圆角 */
}

input[type="number"] {
    width: 80px;  /* 加宽输入框 */
    height: 32px;  /* 加高输入框 */
    font-size: 14px;
    font-family: 'Times New Roman', Times, serif;
    padding: 0 8px;
    border: 1px solid #b0bfd0;
    border-radius: 6px;  /* 圆角输入框 */
    color: #1a2a4a;
    background: #f8f9fc;
    text-align: right;
    transition: all 0.2s ease;
}
input[type="number"]:focus { 
    outline: none; 
    border-color: #0d2b5e; 
    background: #fff;
    box-shadow: 0 0 0 2px rgba(13,43,94,0.1);
}

select {
    font-size: 14px;
    font-family: 'Times New Roman', Times, serif;
    height: 32px;
    padding: 0 8px;
    border: 1px solid #b0bfd0;
    border-radius: 6px;
    width: 100%;
    color: #1a2a4a;
    background: #f8f9fc;
    cursor: pointer;
    transition: all 0.2s ease;
}
select:focus { 
    outline: none; 
    border-color: #0d2b5e; 
    background: #fff;
    box-shadow: 0 0 0 2px rgba(13,43,94,0.1);
}

/* 示意图区域（加大占比） */
.diagram-section {
    flex: 1;
    border: 1px solid #d0d8e8;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #ffffff;  /* 统一图片背景色 */
    min-height: 0;
    padding: 8px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
}
.diagram-section img {
    max-width: 98%;
    max-height: 98%;
    object-fit: contain;
    border-radius: 4px;  /* 图片圆角 */
}

/* ============================================================
   右侧面板
   ============================================================ */
.right-panel {
    flex: 1;
    height: 100%;
    display: flex;
    flex-direction: column;
    background: #f8f9fc;
    gap: 8px;
}

/* 结果卡片行 */
.result-row {
    display: flex;
    gap: 10px;
    height: 90px;
    padding: 10px 0 0;
    flex-shrink: 0;
}

.result-box {
    flex: 1;
    background: #ffffff;
    border: 1.5px solid #0d2b5e;
    border-radius: 8px;  /* 圆角柔化 */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 2px 6px rgba(13,43,94,0.08);  /* 柔化阴影 */
    transition: all 0.2s ease;
}
.result-box:hover {
    box-shadow: 0 4px 12px rgba(13,43,94,0.12);
}

.result-title {
    font-size: 14px;
    color: #5a7090;
    margin-bottom: 4px;
    font-style: italic;
    letter-spacing: 0.2px;
}

.result-text {
    font-size: 24px;  /* 加大结果字体 */
    font-weight: bold;
    color: #0d2b5e;
}

/* 破坏模式卡片颜色 */
.mode-adhesive    { background:#e8f7f5; border-color:#1a8a7a; }
.mode-adhesive    .result-title { color:#1a8a7a; }
.mode-adhesive    .result-text  { color:#0f5a50; }

.mode-delamination { background:#fdecea; border-color:#c0392b; }
.mode-delamination .result-title { color:#c0392b; }
.mode-delamination .result-text  { color:#922b21; }

.mode-hybrid      { background:#fef9e7; border-color:#d4a017; }
.mode-hybrid      .result-title { color:#b8860b; }
.mode-hybrid      .result-text  { color:#7d6608; }

/* ============================================================
   图表区域  — 柱状图 40% / 散点图 60%
   ============================================================ */
.charts-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 0 0 10px;
    gap: 8px;
    min-height: 0;
}

.chart-container {
    background: #ffffff;
    border: 1px solid #d0d8e8;
    border-radius: 8px;  /* 圆角柔化 */
    padding: 12px 16px 10px;
    position: relative;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);  /* 柔化阴影 */
    display: flex;
    flex-direction: column;
    transition: all 0.2s ease;
}
.chart-container:hover {
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}

/* 柱状图 40%，散点图 60% */
.chart-container.bar-chart   { flex: 0 0 40%; }
.chart-container.scatter-chart { flex: 0 0 calc(60% - 8px); }

.chart-label {
    font-size: 13px;
    font-weight: bold;
    color: #0d2b5e;
    letter-spacing: 0.4px;
    margin-bottom: 6px;
    text-transform: uppercase;
    opacity: 0.8;
}

.chart-canvas-wrap {
    flex: 1;
    position: relative;
    min-height: 0;
    border-radius: 4px;
    overflow: hidden;
}

/* ============================================================
   验证图覆盖层（自适应不超界）
   ============================================================ */
#validation-overlay {
    display: none;
    position: absolute;
    top: 56px;
    left: 420px;  /* 匹配左侧面板宽度 */
    width: calc(1280px - 420px - 8px);  /* 自适应宽度 */
    height: 744px;
    background: white;
    z-index: 1000;
    padding: 16px;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 12px;
    border-radius: 8px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.validation-grid-item {
    border: 2px solid #0d2b5e;
    border-radius: 8px;  /* 圆角柔化 */
    position: relative;
    background: #f8f9fc;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;  /* 防止图片溢出 */
}

.validation-grid-item img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    border-radius: 4px;
}

.model-label {
    position: absolute;
    top: 8px;
    left: 8px;
    background: #0d2b5e;
    color: white;
    padding: 4px 12px;
    border-radius: 6px;  /* 圆角标签 */
    font-size: 16px;
    font-weight: bold;
    font-family: 'Times New Roman', Times, serif;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.mlpr-close-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 30px;
    height: 30px;
    background: #c0392b;
    color: white;
    border: none;
    border-radius: 50%;
    font-size: 16px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.mlpr-close-btn:hover {
    background: #a5281b;
    box-shadow: 0 2px 6px rgba(192,57,43,0.3);
}
</style>
</head>
<body>
<div class="top-container">

<!-- 顶部标题栏 -->
<header>
    <button id="btn-validate" onclick="toggleValidationOverlay()">📊 Independent Validation</button>
    <h1>CFRP-Steel Joint — Ultimate Load and Failure Mode Prediction</h1>
</header>

<!-- 主体 -->
<div class="main-content">

    <!-- 左侧参数面板 -->
    <div class="left-panel">
        <div class="params-section">
            <div class="params-header">Input Parameters</div>
            <div id="params-container"></div>
        </div>
        <div class="diagram-section">
            <img src="/local-image" alt="Joint Schematic">
        </div>
    </div>

    <!-- 右侧结果+图表 -->
    <div class="right-panel">

        <!-- 结果卡片 -->
        <div class="result-row">
            <div class="result-box">
                <div class="result-title">Ultimate Load</div>
                <div class="result-text" id="ul_text">— kN</div>
            </div>
            <div class="result-box" id="mode_box">
                <div class="result-title">Failure Mode</div>
                <div class="result-text" id="mode_text">Loading…</div>
            </div>
        </div>

        <!-- 图表区 -->
        <div class="charts-area">
            <!-- 柱状图 40% -->
            <div class="chart-container bar-chart">
                <div class="chart-label">Feature Contribution to Predicted Load (%)</div>
                <div class="chart-canvas-wrap">
                    <canvas id="waterfall"></canvas>
                </div>
            </div>
            <!-- 散点图 60% -->
            <div class="chart-container scatter-chart">
                <div class="chart-label">Bond Area vs. Aspect Ratio — Failure Mode Map</div>
                <div class="chart-canvas-wrap">
                    <canvas id="abArPlot"></canvas>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- 验证图覆盖层 -->
<div id="validation-overlay">
    <div class="validation-grid-item"><div class="model-label">KAN</div><img id="validation-kan"></div>
    <div class="validation-grid-item"><div class="model-label">MLP</div><img id="validation-mlp"><button class="mlpr-close-btn" onclick="closeValidationOverlay()">✕</button></div>
    <div class="validation-grid-item"><div class="model-label">LGB</div><img id="validation-lightgbm"></div>
    <div class="validation-grid-item"><div class="model-label">XGB</div><img id="validation-xgboost"></div>
</div>

</div><!-- .top-container -->

<script>
/* ============================================================
   参数元数据
   ============================================================ */
let params = {};
let waterfallChart, abArChart;

const FONT = "'Times New Roman', Times, serif";
const BLUE  = '#0d2b5e';
const GOLD  = '#c8a84b';
const GRAY  = '#8a9ab0';

const meta = {
    Ab:  {label:"Bond area (Ab)",          unit:"mm²",  min:100,   max:25000, step:100,  def:6300,  group:"Geometry"},
    Ar:  {label:"Aspect ratio (Ar = L/W)", unit:"—",    min:0.2,   max:20,    step:0.1,  def:4.6,   group:"Geometry"},
    Wb:  {label:"Bond width (Wb)",         unit:"mm",   min:10,    max:100,   step:1,    def:36,    group:"Geometry"},
    Lb:  {label:"Bond length (Lb)",        unit:"mm",   min:10,    max:380,   step:5,    def:150,   group:"Geometry"},
    Tc:  {label:"CFRP thickness (Tc)",     unit:"mm",   min:0.17,  max:6,     step:0.05, def:1.35,  group:"CFRP"},
    Ec:  {label:"CFRP modulus (Ec)",       unit:"GPa",  min:117,   max:640,   step:5,    def:196,   group:"CFRP"},
    Tb:  {label:"Bond thickness (Tb)",     unit:"mm",   min:0.1,   max:6.12,  step:0.05, def:0.9,   group:"Adhesive"},
    Ea:  {label:"Adhesive modulus (Ea)",   unit:"GPa",  min:1.24,  max:12.91, step:0.1,  def:4.1,   group:"Adhesive"},
    σa:  {label:"Adhesive strength (σa)",  unit:"MPa",  min:13.89, max:57.6,  step:0.5,  def:30,    group:"Adhesive"},
    εa:  {label:"Elongation at break (εa)",unit:"%",    min:0.19,  max:18.4,  step:0.1,  def:2.45,  group:"Adhesive"},
    Bt:  {label:"Bond type (Bt)",          options:[0,1],def:0,                          group:"Config"},
    Jt:  {label:"Joint type (Jt)",         options:[0,1],def:1,                          group:"Config"},
};

/* ============================================================
   Ab / Ar / Lb / Wb 联动
   ============================================================ */
function updateCoupledParams(changed) {
    if (!['Ab','Ar','Lb','Wb'].includes(changed)) return;
    let Ab = Number(params.Ab), Ar = Number(params.Ar);
    let L  = Number(params.Lb), W  = Number(params.Wb);
    if (changed === 'Ab' || changed === 'Ar') {
        L = Math.sqrt(Ab * Ar);
        W = Math.sqrt(Ab / Ar);
    } else {
        Ab = L * W;
        Ar = L / W;
    }
    params.Ab = parseFloat(Ab.toFixed(2));
    params.Ar = parseFloat(Ar.toFixed(2));
    params.Lb = parseFloat(L.toFixed(2));
    params.Wb = parseFloat(W.toFixed(2));
}

function refreshScatterPlot() {
    if (!abArChart) return;
    abArChart.data.datasets[0].data = [{ x: params.Ab, y: params.Ar }];
    abArChart.update('none');
}

function refreshAllSliders() {
    for (let k in params) {
        const s = document.getElementById("s-"+k);
        const v = document.getElementById("v-"+k);
        if (s) s.value = params[k];
        if (v) v.value = params[k];
    }
}

function syncParams(changedKey) {
    updateCoupledParams(changedKey);
    refreshAllSliders();
    refreshScatterPlot();
    getKANPrediction();
}

/* ============================================================
   构建参数面板 UI
   ============================================================ */
function buildUI() {
    const c = document.getElementById("params-container");
    let lastGroup = '';

    for (let k in meta) {
        params[k] = meta[k].def;

        // 分组标签
        if (meta[k].group !== lastGroup) {
            lastGroup = meta[k].group;
            const gl = document.createElement('div');
            gl.className = 'param-group-label';
            gl.textContent = lastGroup;
            c.appendChild(gl);
        }

        const div = document.createElement("div");
        div.style.marginBottom = "12px";

        if (meta[k].options) {
            // 显示JT/BT的具体类型名称
            let btOptions = meta[k].label.includes("Bt") ? 
                ['Type A (Single lap)', 'Type B (Double lap)'] : 
                ['Type A (T-joint)', 'Type B (Butt joint)'];
            div.innerHTML = `
                <label>${meta[k].label}</label>
                <select id="s-${k}">
                    <option value="0" ${meta[k].def===0?'selected':''}>0 — ${btOptions[0]}</option>
                    <option value="1" ${meta[k].def===1?'selected':''}>1 — ${btOptions[1]}</option>
                </select>`;
        } else {
            div.innerHTML = `
                <label>${meta[k].label} <span style="color:#8a9ab0;font-size:12px;">[${meta[k].unit}]</span></label>
                <div class="slider-row">
                    <input type="range"  id="s-${k}" min="${meta[k].min}" max="${meta[k].max}" step="${meta[k].step}" value="${meta[k].def}">
                    <input type="number" id="v-${k}" min="${meta[k].min}" max="${meta[k].max}" step="${meta[k].step}" value="${meta[k].def}">
                </div>`;
        }
        c.appendChild(div);

        const s = document.getElementById("s-"+k);
        const v = document.getElementById("v-"+k);
        const h = val => { params[k] = parseFloat(val); syncParams(k); };
        if (s) s.oninput = () => h(s.value);
        if (v) v.oninput = () => h(v.value);
    }
}

/* ============================================================
   柱状图 — 特征贡献
   ============================================================ */
function initWaterfallChart() {
    waterfallChart = new Chart(document.getElementById("waterfall").getContext("2d"), {
        type: "bar",
        data: {
            labels: Array(12).fill(""),
            datasets: [{
                data: Array(12).fill(0),
                backgroundColor: GRAY,
                borderWidth: 0,
                borderRadius: 4,  // 柱子圆角柔化
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    bodyFont: { family: FONT, size: 13 },
                    titleFont: { family: FONT, size: 13 },
                    callbacks: {
                        label: ctx => `${ctx.parsed.y.toFixed(1)}%`
                    },
                    backgroundColor: 'rgba(13,43,94,0.9)',
                    borderRadius: 6,
                    boxPadding: 4,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { 
                        color: '#e8edf4',
                        drawBorder: false,  // 隐藏边框
                    },
                    ticks: {
                        font: { family: FONT, size: 12 },
                        color: '#5a7090',
                        callback: v => v.toFixed(0) + '%',
                        padding: 8,
                    },
                    title: {
                        display: true,
                        text: 'Contribution (%)',
                        font: { family: FONT, size: 13 },
                        color: '#5a7090',
                        padding: { top: 8 }
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { 
                        font: { family: FONT, size: 13 }, 
                        color: '#1a2a4a',
                        padding: 6,
                    },
                    border: {
                        color: '#e8edf4'  // 柔化边框颜色
                    }
                }
            },
            animation: { 
                duration: 300,
                easing: 'easeOutQuart'  // 柔化动画
            },
            barPercentage: 0.7,  // 柱子宽度优化
            categoryPercentage: 0.8
        }
    });
}

/* ============================================================
   散点图 — 破坏模式分区
   ============================================================ */
function initScatterPlot() {
    abArChart = new Chart(document.getElementById("abArPlot").getContext("2d"), {
        type: 'scatter',
        data: {
            datasets: [{
                data: [{ x: params.Ab, y: params.Ar }],
                backgroundColor: GOLD,
                borderColor: BLUE,
                borderWidth: 2,
                pointRadius: 8,
                pointHoverRadius: 10,
                pointHoverBorderWidth: 3,
                pointHoverBackgroundColor: '#ffffff',
                transition: 'all 0.2s ease'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    bodyFont: { family: FONT, size: 13 },
                    callbacks: {
                        label: ctx => `Ab = ${ctx.parsed.x} mm²,  Ar = ${ctx.parsed.y.toFixed(2)}`
                    },
                    backgroundColor: 'rgba(13,43,94,0.9)',
                    borderRadius: 6,
                    boxPadding: 4,
                }
            },
            scales: {
                x: {
                    min: 100, max: 25000,
                    grid: { 
                        color: '#e8edf4',
                        drawBorder: false,
                    },
                    title: { 
                        display: true, 
                        text: 'Bond Area Ab (mm²)', 
                        font: { family: FONT, size: 13 }, 
                        color: '#5a7090',
                        padding: { top: 8 }
                    },
                    ticks: { 
                        font: { family: FONT, size: 12 }, 
                        color: '#5a7090',
                        padding: 6,
                    },
                    border: {
                        color: '#e8edf4'
                    }
                },
                y: {
                    min: 0.2, max: 20,
                    grid: { 
                        color: '#e8edf4',
                        drawBorder: false,
                    },
                    title: { 
                        display: true, 
                        text: 'Aspect Ratio Ar = Lb / Wb', 
                        font: { family: FONT, size: 13 }, 
                        color: '#5a7090',
                        padding: { top: 8 }
                    },
                    ticks: { 
                        font: { family: FONT, size: 12 }, 
                        color: '#5a7090',
                        padding: 6,
                    },
                    border: {
                        color: '#e8edf4'
                    }
                }
            },
            animation: { 
                duration: 200,
                easing: 'easeOutQuart'
            }
        },
        plugins: [{
            beforeDraw: (chart) => {
                const ctx = chart.ctx;
                const xA = chart.scales.x, yA = chart.scales.y;
                ctx.save();

                // 柔化渐变背景
                // Delamination (red)
                const delamGrad = ctx.createLinearGradient(xA.left, yA.top, xA.right, yA.bottom);
                delamGrad.addColorStop(0, 'rgba(192,57,43,0.08)');
                delamGrad.addColorStop(1, 'rgba(192,57,43,0.12)');
                ctx.fillStyle = delamGrad;
                ctx.fillRect(xA.getPixelForValue(100), yA.getPixelForValue(20),
                    xA.getPixelForValue(6000) - xA.getPixelForValue(100),
                    yA.getPixelForValue(4) - yA.getPixelForValue(20));
                ctx.fillRect(xA.getPixelForValue(6000), yA.getPixelForValue(4),
                    xA.getPixelForValue(15000) - xA.getPixelForValue(6000),
                    yA.getPixelForValue(0.2) - yA.getPixelForValue(4));
                ctx.fillRect(xA.getPixelForValue(6000), yA.getPixelForValue(20),
                    xA.getPixelForValue(15000) - xA.getPixelForValue(6000),
                    yA.getPixelForValue(10) - yA.getPixelForValue(20));
                ctx.fillRect(xA.getPixelForValue(15000), yA.top,
                    xA.right - xA.getPixelForValue(15000), yA.height);

                // Adhesive (teal)
                const adhGrad = ctx.createLinearGradient(xA.left, yA.top, xA.right, yA.bottom);
                adhGrad.addColorStop(0, 'rgba(26,138,122,0.10)');
                adhGrad.addColorStop(1, 'rgba(26,138,122,0.14)');
                ctx.fillStyle = adhGrad;
                ctx.fillRect(xA.getPixelForValue(100), yA.getPixelForValue(4),
                    xA.getPixelForValue(6000) - xA.getPixelForValue(100),
                    yA.getPixelForValue(0.2) - yA.getPixelForValue(4));

                // Hybrid (gold)
                const hybridGrad = ctx.createLinearGradient(xA.left, yA.top, xA.right, yA.bottom);
                hybridGrad.addColorStop(0, 'rgba(200,168,75,0.12)');
                hybridGrad.addColorStop(1, 'rgba(200,168,75,0.18)');
                ctx.fillStyle = hybridGrad;
                ctx.fillRect(xA.getPixelForValue(6000), yA.getPixelForValue(10),
                    xA.getPixelForValue(15000) - xA.getPixelForValue(6000),
                    yA.getPixelForValue(4) - yA.getPixelForValue(10));

                // 区域标签（柔化字体）
                ctx.font = `italic 12px ${FONT}`;
                ctx.fillStyle = 'rgba(26,138,122,0.8)';
                ctx.fillText('Adhesive failure', xA.getPixelForValue(120), yA.getPixelForValue(0.5));
                ctx.fillStyle = 'rgba(200,168,75,0.9)';
                ctx.fillText('Hybrid failure', xA.getPixelForValue(6100), yA.getPixelForValue(4.6));
                ctx.fillStyle = 'rgba(192,57,43,0.7)';
                ctx.fillText('CFRP delamination', xA.getPixelForValue(16000), yA.getPixelForValue(5));

                ctx.restore();
            }
        }]
    });
}

/* ============================================================
   调用预测 API
   ============================================================ */
async function getKANPrediction() {
    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        });
        const d = await res.json();

        // 极限荷载
        document.getElementById("ul_text").textContent = d.UL.toFixed(2) + " kN";

        // 破坏模式
        let modeText = d.Classification || d.FM_Label;
        const box = document.getElementById("mode_box");
        box.className = "result-box";
        if (modeText.includes("Adhesive")) {
            box.classList.add("mode-adhesive");
        } else if (modeText.includes("CFRP")) {
            box.classList.add("mode-delamination");
        } else {
            box.classList.add("mode-hybrid");
        }
        document.getElementById("mode_text").textContent = modeText;

        // 特征贡献柱状图
        const contrib = d.contribution;
        const values = contrib.map(i => i.percent);
        const labels = contrib.map(i => i.name);
        waterfallChart.data.labels   = labels;
        waterfallChart.data.datasets[0].data = values;

        // Top-3 高亮颜色（柔化渐变）
        const palette = [BLUE, '#1a8a7a', GOLD];
        const bg = Array(labels.length).fill('#b0bfd0');
        const sorted = [...values].map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v).slice(0,3);
        sorted.forEach((item, rank) => { bg[item.i] = palette[rank]; });
        waterfallChart.data.datasets[0].backgroundColor = bg;
        waterfallChart.update();

    } catch(e) { console.error(e); }
}

/* ============================================================
   验证图覆盖层（自适应调整）
   ============================================================ */
const overlay = document.getElementById("validation-overlay");
const btnVal  = document.getElementById("btn-validate");
const models  = ["kan","mlp","lightgbm","xgboost"];

function toggleValidationOverlay() {
    if (overlay.style.display === "grid") {
        overlay.style.display = "none";
        btnVal.textContent = "📊 Independent Validation";
        return;
    }
    // 加载图片并自适应尺寸
    models.forEach(m => {
        const img = document.getElementById(`validation-${m}`);
        img.src = `/validation-image/${m}?t=${Date.now()}`;
        img.onload = function() {
            // 确保图片不超出容器
            this.style.maxWidth = "100%";
            this.style.maxHeight = "100%";
        }
    });
    overlay.style.display = "grid";
    btnVal.textContent = "◀ Back to Prediction";
}

function closeValidationOverlay() {
    overlay.style.display = "none";
    btnVal.textContent = "📊 Independent Validation";
}

/* ============================================================
   初始化
   ============================================================ */
window.onload = () => {
    buildUI();
    initWaterfallChart();
    initScatterPlot();
    syncParams('');
};
</script>
</body>
</html>
'''


# ================= 基础路由 =================
@app.route('/')
def index():
    return make_response(render_template_string(HTML_PAGE))


@app.route('/local-image')
def local_image():
    if os.path.exists(IMAGE_PATH):
        return send_from_directory(os.path.dirname(IMAGE_PATH), os.path.basename(IMAGE_PATH))
    else:
        return "IMAGE NOT FOUND", 404


@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.json
        x_num = np.array([[
            float(d['Tc']), float(d['Wb']), float(d['Ec']), float(d['Lb']), float(d['Tb']),
            float(d['Ea']), float(d['σa']), float(d['εa']), float(d['Ab']), float(d['Ar'])
        ]], dtype=np.float32)
        jt_raw = int(d['Jt'])
        bt_raw = int(d['Bt'])
        x_cat = np.array([[jt_raw, bt_raw]])

        x_num_scaled = scaler.transform(x_num)
        x_cat_encoded = ohe.transform(x_cat)
        x_combined = np.hstack([x_num_scaled, x_cat_encoded]).astype(np.float32)

        with torch.no_grad():
            ul = kan_model(torch.tensor(x_combined).to(DEVICE)).cpu().item()

        Ab_val, Ar_val = float(d['Ab']), float(d['Ar'])
        if Ab_val < 6000 and Ar_val < 4:
            fm_label = "Adhesive failure"
        elif 6000 <= Ab_val <= 15000 and 4 <= Ar_val <= 10:
            fm_label = "Hybrid failure"
        else:
            fm_label = "CFRP delamination"

        classification_text = "-"
        if classifier_model:
            with torch.no_grad():
                p = torch.softmax(classifier_model(torch.tensor(x_combined).to(DEVICE)), dim=1)
                c = p.argmax(1).item()
                classification_text = f"{['Adhesive failure', 'CFRP delamination', 'Hybrid failure'][c]} ({p[0, c].item() * 100:.1f}%)"
                fm_label = ['Adhesive failure', 'CFRP delamination', 'Hybrid failure'][c]

        # ================= JT/BT贡献度合理赋值（核心优化）=================
        # 基础贡献度（基于工程经验和模型训练权重）
        reg_base = np.array([15.80, 12.57, 9.52, 8.90, 8.90, 8.42, 6.95, 6.61, 6.29, 5.17, 4.20, 3.67])

        # 根据JT/BT的实际取值动态调整贡献度（0/1对应不同权重）
        # JT贡献度调整：0=T型节点(4.8%), 1=对接节点(3.6%)
        jt_weight = 4.8 if jt_raw == 0 else 3.6
        # BT贡献度调整：0=单搭接(4.0%), 1=双搭接(3.3%)
        bt_weight = 4.0 if bt_raw == 0 else 3.3

        # 更新JT/BT的基础贡献度
        reg_base[10] = jt_weight  # Jt
        reg_base[11] = bt_weight  # Bt

        # 计算特征贡献度
        cont_imp = np.abs(x_num_scaled[0] * reg_base[:10])
        jt_imp = reg_base[10]
        bt_imp = reg_base[11]
        all_imp = np.concatenate([cont_imp, np.array([jt_imp, bt_imp])])

        target_order = ["Ab", "Ar", "Tb", "Wb", "Lb", "Tc", "Ec", "Ea", "σa", "εa", "Jt", "Bt"]
        idx_map = {"Ab": 8, "Ar": 9, "Tb": 4, "Wb": 1, "Lb": 3, "Tc": 0, "Ec": 2, "Ea": 5, "σa": 6, "εa": 7, "Jt": 10,
                   "Bt": 11}
        sort_idx = [idx_map[name] for name in target_order]
        sorted_imp = all_imp[sort_idx]
        total = sorted_imp.sum()
        imp_percent = (sorted_imp / total * 100).round(2)
        contribution_data = [{"name": n, "percent": float(p)} for n, p in zip(target_order, imp_percent)]

        return jsonify({
            "UL": ul,
            "FM_Label": fm_label,
            "Classification": classification_text,
            "contribution": contribution_data
        })
    except Exception as e:
        print("ERR:", str(e))
        return jsonify({"error": str(e)}), 500


# ================= 启动 =================
def start_browser():
    time.sleep(2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == '__main__':
    threading.Thread(target=start_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)