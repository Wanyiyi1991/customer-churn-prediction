# ========== app/app.py 完整版（带调试） ==========
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import os
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go

# ===== 页面配置 =====
st.set_page_config(
    page_title="客户流失预测系统",
    page_icon="📊",
    layout="wide"
)

# ===== 加载模型 =====
@st.cache_resource
def load_artifacts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 列出当前目录所有文件（调试用）
    files_in_dir = os.listdir(current_dir)
    st.write(f"📁 app/ 目录下的文件: {files_in_dir}")
    
    # 加载预处理器
    preprocessor_path = os.path.join(current_dir, 'preprocessor.pkl.gz')
    st.write(f"🔍 预处理器路径: {preprocessor_path}")
    st.write(f"   文件存在: {os.path.exists(preprocessor_path)}")
    
    with gzip.open(preprocessor_path, 'rb') as f:
        preprocessor = joblib.load(f)
    
    # 加载模型
    model_path = os.path.join(current_dir, 'churn_model.json')
    st.write(f"🔍 模型路径: {model_path}")
    st.write(f"   文件存在: {os.path.exists(model_path)}")
    
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 加载特征名
    feature_path = os.path.join(current_dir, 'feature_names.pkl')
    st.write(f"🔍 特征名路径: {feature_path}")
    st.write(f"   文件存在: {os.path.exists(feature_path)}")
    
    feature_names = joblib.load(feature_path)
    
    # 打印预处理器期望的输入
    st.write(f"📋 预处理器期望的列: {list(preprocessor.feature_names_in_)}")
    
    explainer = shap.TreeExplainer(model)
    
    return preprocessor, model, feature_names, explainer

# 加载
preprocessor, model, feature_names, explainer = load_artifacts()

# ===== 标题 =====
st.title("📊 电信客户流失预测系统")
st.markdown("---")

# ===== 侧边栏 =====
st.sidebar.header("📝 输入客户信息")

col1, col2 = st.sidebar.columns(2)

with col1:
    gender = st.selectbox("性别", ["Female", "Male"])
    senior_citizen = st.selectbox("是否老年人", [0, 1])
    partner = st.selectbox("是否有伴侣", ["Yes", "No"])
    dependents = st.selectbox("是否有家属", ["No", "Yes"])
    tenure = st.slider("在网时长（月）", 0, 72, 12)
    phone_service = st.selectbox("是否开通电话", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("是否多条线路", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("互联网服务类型", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("在线安全", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("在线备份", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("设备保护", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("技术支持", ["No", "Yes", "No internet service"])

streaming_tv = st.sidebar.selectbox("流媒体电视", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("流媒体电影", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("合同类型", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("电子账单", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "支付方式",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.sidebar.number_input("月费 ($)", 18.0, 120.0, 70.0, 0.5)
total_charges = st.sidebar.number_input("总消费额 ($)", 0.0, 9000.0, 840.0, 0.5)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 预测流失概率", use_container_width=True, type="primary")

# ===== 预测 =====
if predict_btn:
    # 构建输入
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': int(senior_citizen),
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(total_charges)
    }])
    
    # 确保列顺序
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    input_data = input_data[feature_order]
    
    # 打印调试信息
    st.write("📝 输入数据:")
    st.write(input_data)
    st.write("📝 输入数据类型:")
    st.write(input_data.dtypes)
    
    try:
        input_processed = preprocessor.transform(input_data)
        st.write(f"✅ 转换成功！输出形状: {input_processed.shape}")
        
        churn_prob = model.predict_proba(input_processed)[0, 1]
        shap_values = explainer.shap_values(input_processed)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("流失概率", f"{churn_prob:.1%}")
        with col2:
            risk = "🔴 高风险" if churn_prob > 0.7 else ("🟡 中风险" if churn_prob > 0.3 else "🟢 低风险")
            st.metric("风险等级", risk)
        with col3:
            st.metric("预期月损失 ($)", f"${monthly_charges * churn_prob:.2f}")
        
        st.markdown("---")
        st.subheader("🔍 SHAP解释")
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(values=shap_values[0], base_values=explainer.expected_value,
                           data=input_processed[0], feature_names=feature_names),
            show=False, max_display=8
        )
        st.pyplot(fig_shap)
        
    except Exception as e:
        st.error(f"❌ 转换失败: {str(e)}")
        st.write("完整错误类型:", type(e).__name__)

else:
    st.info("👈 请在左侧输入客户信息，然后点击「预测流失概率」按钮")
st.markdown("---")
st.caption("数据来源: Telco Customer Churn (Kaggle)")
