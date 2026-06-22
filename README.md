# 📊 客户流失预测与干预系统

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-red)](https://github.com/shap/shap)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 端到端机器学习项目：从7,043条电信客户数据出发，经历数据清洗→类别不平衡处理→多模型对比→超参数调优→SHAP解释→客户分层→Streamlit部署，构建完整的流失预警与干预系统。

## 📊 项目概览

扮演电信公司数据科学家，构建客户流失预测模型并部署为可交互的Web应用。项目覆盖机器学习的完整生命周期：**数据预处理、类别不平衡处理（SMOTE）、4个模型对比、Grid Search调优、SHAP模型解释、客户风险分层、干预策略设计、Streamlit部署**。

## ❓ 回答的核心问题

| 序号 | 分析步骤 | 回答的问题 |
|------|----------|-----------|
| 1 | EDA | 哪些客户特征与流失最相关？ |
| 2 | 类别不平衡处理 | 如何解决仅26.5%流失客户的样本偏斜？ |
| 3 | 多模型对比 | 哪个模型预测流失最准？ |
| 4 | SHAP解释 | 什么因素驱动流失？影响多大？ |
| 5 | 客户分层 | 如何将客户分成高/中/低风险？ |
| 6 | 干预策略 | 对不同风险层该采取什么行动？ |
| 7 | Streamlit部署 | 如何让业务方直接使用模型？ |

## 🔍 核心发现

1. **合同类型是流失的最强信号**：月合同客户流失率42.7%，年合同仅2.8%（差15倍）
2. **短在网+高月费=极度危险**：在网<12个月且月费>$70的客户是流失重灾区
3. **增值服务缺失推高流失**：没有OnlineSecurity/TechSupport的客户流失率显著更高
4. **SHAP量化了每个因素的影响**：合同类型、在网时长、月费是Top3驱动因素
5. **20%的高风险客户贡献了60%+的流失**：精准识别后干预效率极高

## 🛠 技术栈

- **数据清洗**: Pandas, NumPy
- **预处理**: Scikit-learn Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler
- **类别不平衡**: imbalanced-learn SMOTE
- **建模**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **调优**: GridSearchCV, StratifiedKFold
- **解释**: SHAP (Summary/Dependence/Waterfall Plot)
- **部署**: Streamlit, Joblib
- **可视化**: Matplotlib, Seaborn, Plotly

## 📁 项目结构
customer-churn-prediction/  
├── README.md  
├── requirements.txt  
├── data/  
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv  
├── notebooks/  
│ └── customer-churn-prediction.ipynb # 完整分析过程  
├── images/  
│ ├── churn_distribution.png  
│ ├── contract_churn.png  
│ ├── numerical_distributions.png  
│ ├── categorical_churn_rates.png  
│ ├── churn_correlation.png  
│ ├── tenure_vs_charges.png  
│ ├── model_comparison.png  
│ ├── model_diagnosis.png  
│ ├── shap_summary.png  
│ ├── shap_bar.png  
│ ├── shap_dependence.png  
│ ├── shap_waterfall.png  
│ └── risk_stratification.png  
├── app/  
│ ├── app.py # Streamlit应用  
│ ├── preprocessor.pkl.gz # 预处理器  
│ ├── churn_model.pkl.gz # 训练好的模型  
│ └── feature_names.pkl # 特征名  
└── report/  
└── 客户流失预测与干预系统报告.md


## 🚀 快速复现

```bash
# 1. 克隆仓库
git clone https://github.com/Wanyiyi1991/customer-churn-prediction.git
cd customer-churn-prediction

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载数据
# 从 Kaggle 搜索 "Telco Customer Churn" 下载CSV文件
# 放入 data/ 文件夹

# 4. 运行完整分析
jupyter notebook notebooks/churn_analysis.ipynb

# 5. 启动交互式应用
streamlit run app/app.py

## 在线演示

([https://static.streamlit.io/badges/streamlit_badge_black_white.svg](https://llzkam4wbrxgnappsjohccj.streamlit.app/)(我的Streamlit Cloud链接)

> 在左侧面板输入客户特征，点击预测，即可查看流失概率、SHAP解释和个性化干预建议。

## 📈 关键可视化

### SHAP特征重要性——什么因素最影响流失？

[https://images/shap_summary.png](https://github.com/Wanyiyi1991/customer-churn-prediction/blob/main/images/shap_summary.png)

> 合同类型(Month-to-month)是流失的最强正向驱动因素。高月费、短在网时间紧随其后。

### 客户流失风险分布图

[[https://images/tenure_vs_charges.png](https://images/tenure_vs_charges.png)](https://github.com/Wanyiyi1991/customer-churn-prediction/blob/main/images/tenure_vs_charges.png)

> 左上角橙色密集区：在网<12个月 + 月费>$70的客户，流失率极高。这是干预的优先级最高区域。

### 模型诊断——混淆矩阵与ROC曲线

[[https://images/model_diagnosis.png](https://images/model_diagnosis.png)](https://github.com/Wanyiyi1991/customer-churn-prediction/blob/main/images/model_diagnosis.png)

> 最佳模型在测试集上ROC AUC达0.84+，Recall 0.65+，能以较高准确率识别即将流失的客户。

## 💡 业务建议与干预策略

### 客户三层分级

|风险层|占比|实际流失率|干预策略|
|---|---|---|---|
|🟢 低风险|~50%|<5%|常规维护|
|🟡 中风险|~30%|15-30%|个性化推荐 + 满意度调研|
|🔴 高风险|~20%|>60%|**立即触达**：推荐年合同 + 赠送安全服务|

### 具体行动

1. **月合同→年合同转化活动**：年合同8折优惠 + 免费OnlineSecurity一个月
    
2. **增值服务交叉销售**：高月费客户捆绑销售设备保护+技术支持
    
3. **新客户90天关怀计划**：在网<6个月的客户，每30天主动触达
    

### 预计商业影响

- 高风险客户约占20%，如能挽留其中30%，预计可挽回年收入~$447,000
    
- 模型可部署为月度自动化评分系统，持续监控并触发干预
    

## 👤 关于我

[一个数据分析探索者、爱好者！
QQ：783899056
Email：wanyiyi1991@gmail.com]

---

⭐ 如果这个项目对你有帮助，欢迎给个Star！
