# ========== app/app.py ==========
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go
import os

# ===== 页面配置 =====
st.set_page_config(
    page_title="客户流失预测系统",
    page_icon="📊",
    layout="wide"
)

# ===== 加载模型（自动判断文件格式） =====
@st.cache_resource
def load_artifacts():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载预处理器
    preprocessor_path = os.path.join(current_dir, 'preprocessor.pkl.gz')
    if not os.path.exists(preprocessor_path):
        st.error(f"找不到文件: {preprocessor_path}")
        st.stop()
    
    import gzip
    with gzip.open(preprocessor_path, 'rb') as f:
        preprocessor = joblib.load(f)
    
    # 加载模型（优先找.json，其次找.pkl.gz）
    model_json = os.path.join(current_dir, 'churn_model.json')
    model_pkl = os.path.join(current_dir, 'churn_model.pkl.gz')
    
    if os.path.exists(model_json):
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_json)
    elif os.path.exists(model_pkl):
        with gzip.open(model_pkl, 'rb') as f:
            model = joblib.load(f)
    else:
        st.error("找不到模型文件！请确保 churn_model.json 或 churn_model.pkl.gz 存在")
        st.stop()
    
    # 加载特征名
    feature_path = os.path.join(current_dir, 'feature_names.pkl')
    if os.path.exists(feature_path):
        feature_names = joblib.load(feature_path)
    else:
        st.error("找不到 feature_names.pkl")
        st.stop()
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    return preprocessor, model, feature_names, explainer

# 加载
preprocessor, model, feature_names, explainer = load_artifacts()

# ===== 标题 =====
st.title("📊 电信客户流失预测系统")
st.markdown("基于XGBoost + SMOTE，ROC AUC 0.85 | 输入客户信息预测流失概率")
st.markdown("---")

# ===== 侧边栏：用户输入 =====
st.sidebar.header("📝 输入客户信息")

col1, col2 = st.sidebar.columns(2)

with col1:
    gender = st.selectbox("性别", ["Female", "Male"])
    senior_citizen = st.selectbox("是否老年人（0=否, 1=是）", [0, 1])
    partner = st.selectbox("是否有伴侣", ["No", "Yes"])
    dependents = st.selectbox("是否有家属", ["No", "Yes"])
    tenure = st.slider("在网时长（月）", 0, 72, 12)
    phone_service = st.selectbox("是否开通电话", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("是否多条线路", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("互联网服务类型", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("在线安全", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("在线备份", ["No", "Yes", "No internet service"])
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
total_charges = st.sidebar.number_input("总消费额 ($)", 0.0, 9000.0, monthly_charges * tenure, 0.5)

# ===== 预测按钮 =====
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔮 预测流失概率", use_container_width=True, type="primary")

# ===== 主面板 =====
if predict_btn:
    # 构建输入数据
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': int(senior_citizen),      # 转整数
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),                     # 转整数
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
        'MonthlyCharges': float(monthly_charges),  # 转浮点数
        'TotalCharges': float(total_charges)       # 转浮点数
    }])
    
    # 确保列顺序与训练时完全一致
    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]
    input_data = input_data[feature_order]
    
    # 预处理
    input_processed = preprocessor.transform(input_data)
    
    # 预测
    churn_prob = model.predict_proba(input_processed)[0, 1]
    
    # SHAP值
    shap_values = explainer.shap_values(input_processed)
    
    # ===== 结果展示 =====
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "流失概率",
            f"{churn_prob:.1%}",
            delta="高风险" if churn_prob > 0.5 else "低风险",
            delta_color="inverse"
        )
    
    with col2:
        risk_level = "🔴 高风险" if churn_prob > 0.7 else ("🟡 中风险" if churn_prob > 0.3 else "🟢 低风险")
        st.metric("风险等级", risk_level)
    
    with col3:
        st.metric("预期月损失 ($)", f"${monthly_charges * churn_prob:.2f}")
    
    st.markdown("---")
    
    # ===== SHAP解释图 =====
    st.subheader("🔍 关键影响因素（SHAP）")
    
    fig_shap, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_processed[0],
            feature_names=feature_names
        ),
        show=False,
        max_display=8
    )
    st.pyplot(fig_shap)
    
    # ===== 干预建议 =====
    st.markdown("---")
    st.subheader("💡 个性化干预建议")
    
    suggestions = []
    if contract == "Month-to-month" and churn_prob > 0.5:
        suggestions.append("🔴 **核心建议**：月度合同+高流失风险。推荐年合同8折优惠 + 赠送OnlineSecurity。")
    if online_security == "No" and churn_prob > 0.3:
        suggestions.append("🟡 推荐开通OnlineSecurity——SHAP分析显示这是降低流失的关键因子。")
    if tenure < 12 and churn_prob > 0.3:
        suggestions.append("🟡 新客户(<12个月)：纳入90天关怀计划，每30天主动触达。")
    if tech_support == "No" and churn_prob > 0.3:
        suggestions.append("🟡 推荐开通TechSupport——缺乏技术支持的客户流失率显著更高。")
    if monthly_charges > 80 and churn_prob > 0.3:
        suggestions.append("🟠 高月费客户：提供忠诚度折扣或增值服务捆绑包。")
    
    if not suggestions:
        suggestions.append("✅ 该客户风https://github.com/Wanyiyi1991/customer-churn-prediction低，维持常规维护策略即可。")
    
    for s in suggestions:
        st.markdown(s)

else:
    st.info("👈 请在左侧输入客户信息，然后点击「预测流失概率」按钮")

# ===== 页脚 =====
st.markdown("---")
st.caption("🔗 完整分析代码见 [GitHub](链接) | 数据来源: Telco Customer Churn (Kaggle)")