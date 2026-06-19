# app.py —— 客户流失预测交互式看板
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
import matplotlib.pyplot as plt
import shap
import plotly.express as px
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
    # 加载压缩的预处理器
    with gzip.open('preprocessor.pkl.gz', 'rb') as f:
        preprocessor = joblib.load(f)
    
    # 加载压缩的模型
    with gzip.open('churn_model.pkl.gz', 'rb') as f:
        model = joblib.load(f)
    
    # 加载特征名
    feature_names = joblib.load('feature_names.pkl')
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    
    return preprocessor, model, feature_names, explainer

preprocessor, model, feature_names, explainer = load_artifacts()

# ===== 标题 =====
st.title("📊 电信客户流失预测系统")
st.markdown("---")

# ===== 侧边栏：用户输入 =====
st.sidebar.header("📝 输入客户信息")

col1, col2 = st.sidebar.columns(2)

with col1:
    gender = st.selectbox("性别", ["Female", "Male"])
    senior_citizen = st.selectbox("是否老年人", ["No", "Yes"])
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
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
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
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])
    
    # 预处理
    input_processed = preprocessor.transform(input_data)
    
    # 预测
    churn_prob = model.predict_proba(input_processed)[0, 1]
    churn_pred = model.predict(input_processed)[0]
    
    # 计算SHAP值
    shap_values = explainer.shap_values(input_processed)
    
    # ===== 结果展示 =====
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "流失概率",
            f"{churn_prob:.1%}",
            delta="高风险" if churn_prob > 0.5 else "低风险",
            delta_color="inverse" if churn_prob > 0.5 else "normal"
        )
    
    with col2:
        risk_level = "🔴 高风险" if churn_prob > 0.7 else ("🟡 中风险" if churn_prob > 0.3 else "🟢 低风险")
        st.metric("风险等级", risk_level)
    
    with col3:
        monthly_risk = monthly_charges * churn_prob
        st.metric("预期月损失 ($)", f"${monthly_risk:.2f}")
    
    st.markdown("---")
    
    # ===== 可视化 =====
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 流失概率仪表盘")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "流失概率 (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#F18F01"},
                'steps': [
                    {'range': [0, 30], 'color': "#2E86AB"},
                    {'range': [30, 70], 'color': "#F18F01"},
                    {'range': [70, 100], 'color': "#C73E1D"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': churn_prob * 100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_right:
        st.subheader("🔍 关键影响因素（SHAP）")
        fig_shap, ax = plt.subplots(figsize=(8, 5))
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
    
    st.markdown("---")
    
    # ===== 干预建议 =====
    st.subheader("💡 个性化干预建议")
    
    suggestions = []
    if contract == "Month-to-month":
        suggestions.append("🔴 **核心建议**：该客户使用月合同，流失风险极高。推荐年合同8折优惠 + 赠送OnlineSecurity一个月。")
    if online_security == "No":
        suggestions.append("🟡 推荐开通OnlineSecurity服务——SHAP分析显示这是降低流失风险的关键因子。")
    if tenure < 12:
        suggestions.append("🟡 新客户(<12个月)：纳入90天关怀计划，每30天主动触达。")
    if tech_support == "No":
        suggestions.append("🟡 推荐开通TechSupport——缺乏技术支持的客户流失率显著更高。")
    if monthly_charges > 80:
        suggestions.append("🟠 高月费客户：提供忠诚度折扣或增值服务捆绑包。")
    
    if not suggestions:
        suggestions.append("✅ 该客户风险较低，维持常规维护策略即可。")
    
    for s in suggestions:
        st.markdown(s)

else:
    # 初始状态
    st.info("👈 请在左侧输入客户信息，然后点击「预测流失概率」按钮")
    
    # 展示示例数据
    st.subheader("📋 使用说明")
    st.markdown("""
    本系统基于7,043条电信客户数据训练，使用XGBoost + SMOTE模型，ROC AUC达到0.84+。
    
    **如何使用：**
    1. 在左侧面板输入客户特征
    2. 点击「预测流失概率」
    3. 查看预测结果、SHAP解释和干预建议
    
    **典型高风险客户画像：**
    - 月度合同 + 高月费(>$70) + 在网<12个月 + 无在线安全/技术支持
    """)

# ===== 页脚 =====
st.markdown("---")
st.caption("🔗 完整分析代码见 [GitHub](https://github.com/Wanyiyi1991/customer-churn-prediction) | 数据来源: Telco Customer Churn (Kaggle)")