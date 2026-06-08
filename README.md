# 📚 本地知识库问答系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-0.1+-green.svg)](https://ollama.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> 一个完全本地运行、基于自然语言的智能数据分析助手。  
> 上传 Excel，说话即可完成**脱敏、统计、图表、关联、回归、聚类**等操作。  
> 所有数据留在您自己的电脑上，隐私零风险。

---

## ✨ 功能特色

### 📂 数据管理
- 上传 `.xlsx` / `.xls` 文件，自动生成**数据质量报告**（缺失值、重复值、异常值）
- 支持多文件切换，同时处理多个表格

### 🔐 隐私保护
- 自动识别敏感列（学号、身份证、手机号、邮箱等）并脱敏（保留后四位）
- 一键导出脱敏后的完整 CSV 文件

### 📊 数据分析与统计
- **排序取前N名**：例如“前10名学生”
- **分组统计**：例如“统计各班男女比例”
- **自然语言生成图表**：柱状图、折线图、饼图、直方图

### 🔗 多表关联
- 支持 `inner` / `left` / `right` / `outer` 连接
- 交互式选择关联键，无需编写 SQL

### 🧠 高级统计分析
- **相关系数热力图**
- **线性回归分析**（散点图 + 回归线）
- **KMeans 聚类**（可自定义聚类数量）

### 💬 自然语言问答
- 基于本地大模型（Ollama + qwen2.5:7b）理解意图并生成分析报告
- 完全离线，不依赖任何云 API

### 📥 结果导出
- 脱敏结果、统计表格、关联结果等均可一键导出 CSV

### 🗂️ 其他
- 查询历史与收藏（保存常用提问）
- 多轮对话上下文记忆（可开关）

---

## 🖥️ 技术栈

| 类别         | 技术                                      |
| ------------ | ----------------------------------------- |
| 前端/界面    | Streamlit                                 |
| 数据处理     | pandas, numpy                             |
| 统计分析     | scikit-learn（线性回归、KMeans）           |
| 可视化       | matplotlib                                |
| 大模型       | Ollama + qwen2.5:7b（可替换其他本地模型）  |
| 本地存储     | JSON（历史与收藏）                         |

> 本版本不依赖向量数据库（避免启动卡顿），保留了所有 Excel 核心功能。

---

## 📦 安装与使用
 1. 克隆仓库
git clone https://github.com/your-username/local-knowledge-qa-system.git
cd local-knowledge-qa-system

2. 安装依赖
pip install -r requirements.txt

3. 启动 Ollama 服务（需提前安装 Ollama）
ollama serve
# 另开一个终端拉取模型
ollama pull qwen2.5:7b

4. 运行 Streamlit 应用
streamlit run app.py
浏览器会自动打开 http://localhost:8501。

5. 上传 Excel 文件，开始提问！
示例问题：
将学号脱敏，只显示后四位
前20名学生
统计各班男女比例
画柱状图（自动识别分组列和数值列）
生成索引表（提取序号、姓名、班级、学号、专业）

🗂️ 项目结构
text
.
├── app.py                  # 主程序
├── requirements.txt        # Python 依赖
├── query_history.json      # 历史与收藏（自动生成）
├── README.md               # 本文件
└── LICENSE                 # MIT 许可证

🔧 常见问题
❓ 上传文件后没有反应？
请确保文件为 .xlsx 或 .xls 格式，且至少有一行数据。
若文件过大（>10万行），首次加载可能稍慢，请稍等。

❓ 脱敏不彻底怎么办？
系统会识别列名中含 学号、手机、身份证、邮箱 的列。
如果您的列名不同，可以修改 mask_sensitive_columns 函数中的关键词列表。

❓ 图表无法生成？
请在提问时明确分组列和数值列，例如“按班级画柱状图”。
系统会尝试自动识别，但更精确的描述能提高成功率。

❓ 想使用其他大模型？
修改 app.py 中的 model="qwen2.5:7b" 为其他 Ollama 支持的模型名（如 llama3:8b, mistral:7b）。

❓ 导出的 CSV 是乱码？
系统默认使用 UTF-8 编码，可直接用 Excel 打开。若出现乱码，请用记事本打开后另存为 UTF-8 格式。

🚀 未来计划
支持 PDF / Word 文档的语义检索（可选轻量级向量库）

增加数据透视表自动生成

更多图表类型（箱线图、散点图矩阵）

支持 SQL 查询模式

Web 端直接编辑单元格并重新分析

🤝 贡献
欢迎提交 Issue 和 Pull Request。如有建议或 bug，请告诉我！
