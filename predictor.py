#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image
from decimal import Decimal, ROUND_HALF_UP  # 新增：精准控制小数精度

# 简化字体配置（仅解决负号显示，无需中文）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 加载模型
model = joblib.load('GBD.pkl')

# 特征名：医学通用英文缩写
feature_names = [
    "Gender", "Age", "BMI", "TG", "LDL-C", 
    "HDL-C", "ALT", "AST/ALT", "TP", "ALB", 
    "Scr", "UA", "FBG", "WBC", "LYM#", 
    "MCH", "PLT"
]

# StreamLit界面（输入框保留中文+英文缩写）
st.title("脂肪肝预测器")

# 输入框
年龄 = st.number_input("年龄(Age):", min_value=0, max_value=120, value=0)
性别 = st.selectbox("性别(Gender):", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
体质指数 = st.number_input("体质指数(BMI):", min_value=0, max_value=30, value=0)
甘油三酯 = st.number_input("甘油三酯(TG):", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
低密度脂蛋白胆固醇 = st.number_input("低密度脂蛋白胆固醇(LDL-C):", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
高密度脂蛋白胆固醇 = st.number_input("高密度脂蛋白胆固醇(HDL-C):", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
谷丙转氨酶 = st.number_input("谷丙转氨酶(ALT):", min_value=0, max_value=500, value=0, step=1)
谷草酶谷丙酶 = st.number_input("谷草酶谷丙酶(AST/ALT):", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
总蛋白 = st.number_input("总蛋白(TP):", min_value=0, max_value=100, value=0, step=1)
白蛋白 = st.number_input("白蛋白(ALB):", min_value=0, max_value=70, value=0, step=1)
血肌酐 = st.number_input("血肌酐(Scr):", min_value=0, max_value=500, value=0, step=1)
血尿酸 = st.number_input("血尿酸(UA):", min_value=0, max_value=1000, value=0, step=1)
空腹血糖 = st.number_input("空腹血糖(FBG):", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
白细胞 = st.number_input("白细胞(WBC):", min_value=0.0, max_value=30.0, value=0.0, step=0.1)
淋巴细胞计数 = st.number_input("淋巴细胞计数(LYM#):", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
平均血红蛋白 = st.number_input("平均血红蛋白(MCH):", min_value=0, max_value=50, value=0, step=1)
血小板 = st.number_input("血小板(PLT):", min_value=0, max_value=1000, value=0, step=1)

# 处理输入数据（核心：用Decimal精准控制2位小数）
feature_values = [
    性别,年龄,体质指数,甘油三酯,低密度脂蛋白胆固醇,高密度脂蛋白胆固醇,
    谷丙转氨酶,谷草酶谷丙酶,总蛋白,白蛋白,血肌酐,血尿酸,空腹血糖,
    白细胞,淋巴细胞计数,平均血红蛋白,血小板
]  
# 彻底解决浮点误差：用Decimal保留2位小数
feature_values = [
    float(Decimal(str(x)).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)) 
    for x in feature_values
]
# 转为numpy数组（供模型预测）
features = np.array([feature_values], dtype=np.float32)

# 构建两个DataFrame：
# 1. 模型预测用（数值型，精准）
features_df = pd.DataFrame(features, columns=feature_names, dtype=np.float32)
# 2. SHAP显示用（字符串型，格式化后无超长小数）
formatted_features = {col: f"{val:.2f}" for col, val in zip(feature_names, feature_values)}
formatted_values = [formatted_features[col] for col in feature_names]
features_df_display = pd.DataFrame([formatted_values], columns=feature_names)

# 预测逻辑
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class} (1: 有脂肪肝, 0: 无脂肪肝)")
    st.write(f"**预测概率:** {predicted_proba}")

    # 生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据模型预测，你有较高的脂肪肝风险。"
            f"模型预测你患脂肪肝的概率为 {probability:.1f}%。"
            "建议及时咨询医生，进行进一步检查和干预。"
        )
    else:
        advice = (
            f"根据模型预测，你患脂肪肝的风险较低。"
            f"模型预测你无脂肪肝的概率为 {probability:.1f}%。"
            "建议保持健康的生活方式，并定期进行体检。"
        )
    st.write(advice)

    # ========== SHAP图（使用格式化后的DataFrame，数值显示为2位小数） ==========
    st.subheader("预测结果解释（SHAP Force Plot）")
    plt.clf()
    plt.close('all')
    
    # 计算SHAP值（用模型预测用的features_df）
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    # 生成SHAP Force Plot（用显示用的features_df_display）
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        features_df_display.iloc[0],  # 关键：用格式化后的字符串数值
        feature_names=feature_names,
        out_names="Fatty Liver Probability",
        show=False,
        matplotlib=True,
        figsize=(12, 4)
    )
    plt.tight_layout()
    
    # 保存并显示图片
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    st.image(img, use_column_width=True)
    plt.close('all')
    
    # 特征缩写对照表
    st.subheader("特征缩写对照表")
    abbr_map = {
        "Gender": "性别", "Age": "年龄", "BMI": "体质指数", "TG": "甘油三酯", 
        "LDL-C": "低密度脂蛋白胆固醇", "HDL-C": "高密度脂蛋白胆固醇", 
        "ALT": "谷丙转氨酶", "AST/ALT": "谷草酶谷丙酶", "TP": "总蛋白", 
        "ALB": "白蛋白", "Scr": "血肌酐", "UA": "血尿酸", "FBG": "空腹血糖", 
        "WBC": "白细胞", "LYM#": "淋巴细胞计数", "MCH": "平均血红蛋白", 
        "PLT": "血小板"
    }
    abbr_df = pd.DataFrame({
        "英文缩写": list(abbr_map.keys()),
        "中文含义": list(abbr_map.values())
    })
    st.dataframe(abbr_df, use_container_width=True)
    
    # SHAP图说明
    st.write("""
    **SHAP Force Plot说明：**
    - Red features: Increase the probability of fatty liver (红色特征：增加脂肪肝概率)；
    - Blue features: Decrease the probability of fatty liver (蓝色特征：降低脂肪肝概率)；
    - Feature bar length: The degree of impact on prediction results (特征条长度：对预测结果的影响程度)。
    """)

