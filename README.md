一·#1. app.py【项目主程序】
整套Flask 后端 + 内嵌 HTML 前端源码，是整个预测软件入口：
后端：加载 KAN 回归 / 分类模型、数据集、标准化器，接收前端滑块参数，完成极限承载力Ultimate Load计算、失效模式分类、12 项特征贡献度权重计算；
前端：内置完整 CSS+Chart.js，实现左侧参数滑块面板（几何 / CFRP / 胶粘剂 / 节点类型四大分组）、右侧双结果卡片 + 特征贡献柱状图 + 失效分区散点图、左上角独立验证弹窗按钮，和你成品界面 1:1 匹配；
路由：图片访问路由、预测接口、验证图片路由，本地python app.py自动打开http://127.0.0.1:5000网页。
二、数据文件
data.csv
模型训练原始数据集，程序启动时一次性读取，用来拟合 MinMaxScaler、OneHotEncoder，完成特征缩放器初始化（解决缩放器未拟合报错），包含Tc/Wb/Ec/Lb/Ab/Ar/Jt/Bt等全部输入特征字段。
三、模型权重文件（pkl）
regression_KAN_model.pkl：回归 KAN 模型，输入 12 维特征，输出接头极限抗拉承载力Ultimate Load(kN)；
classification_KAN_model.pkl：分类 KAN 模型，三分类任务：Adhesive failure(粘接破坏)/Hybrid failure(混合破坏)/CFRP delamination(剥离破坏)，输出失效类型 + 分类置信百分比。
四、图片资源（img 目录图片，当前平铺在根目录）
GUI 的图A.png：左侧参数面板底部CFRP - 钢接头结构示意图，页面通过/local-image路由加载展示；
KAN_optimized.png / MLP_optimized / LightGBM_optimized / XGBoost_optimized.png：四种算法独立试验验证对比图，点击页面左上角Independent Validation按钮，弹窗 2×2 布局展示四张对比图。
五、配置 & 说明文档
Requirements.txt：项目依赖清单，云端部署（PythonAnywhere/Render）一键批量安装flask、torch、numpy、pandas、scikit-learn；
README.md：项目说明文档，介绍项目用途：基于 KAN 神经网络的 CFRP-steel joints 接头力学性能可视化预测平台，可自定义全参数、实时预测承载力与失效形式。
