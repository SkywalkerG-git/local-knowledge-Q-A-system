import os
import re
import json
import datetime
import time
import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from io import BytesIO

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "query_history.json")

st.set_page_config(page_title="本地知识库问答系统", layout="wide")
st.title("📚 本地知识库问答系统")
st.caption("支持Excel多表关联、图表生成、统计分析、隐私保护 | 自动质量报告 | 查询历史收藏 | 完全本地运行")

# ----------------------------- 数据脱敏 -----------------------------
def mask_sensitive_columns(df):
    df = df.copy()
    sensitive_keywords = ['学号', '身份证', '手机', '电话', '邮箱', 'email']
    for col in df.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in sensitive_keywords):
            df[col] = df[col].astype(str).apply(
                lambda x: '*' * (len(x)-4) + x[-4:] if len(x) > 4 else x
            )
    return df

# ----------------------------- 查询历史与收藏 -----------------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"history": [], "favorites": []}

def save_history(data):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_to_history(query, answer):
    data = load_history()
    data["history"].insert(0, {"query": query, "answer": answer[:200], "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    if len(data["history"]) > 50:
        data["history"] = data["history"][:50]
    save_history(data)

def add_to_favorites(query, answer):
    data = load_history()
    fav = {"query": query, "answer": answer[:200], "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if fav not in data["favorites"]:
        data["favorites"].append(fav)
        save_history(data)
        return True
    return False

def remove_from_favorites(index):
    data = load_history()
    if 0 <= index < len(data["favorites"]):
        del data["favorites"][index]
        save_history(data)

# ----------------------------- 高级统计分析 -----------------------------
def compute_correlation(df):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty:
        return None, None
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    plt.colorbar(im)
    plt.title("相关系数矩阵热力图")
    return corr, fig

def linear_regression_analysis(df, x_col, y_col):
    if x_col not in df.columns or y_col not in df.columns:
        return None, None, "列名不存在"
    df_clean = df[[x_col, y_col]].dropna()
    if len(df_clean) < 2:
        return None, None, "数据点不足"
    X = df_clean[x_col].values.reshape(-1, 1)
    y = df_clean[y_col].values
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, alpha=0.6, label='实际值')
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    ax.plot(x_range, y_pred, 'r', label=f'回归线 (y={model.coef_[0]:.2f}x+{model.intercept_:.2f})')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"线性回归: {y_col} ~ {x_col}  (R2={r2:.3f})")
    ax.legend()
    return model, fig, f"R2 = {r2:.3f}, 斜率 = {model.coef_[0]:.3f}, 截距 = {model.intercept_:.3f}"

def kmeans_clustering(df, n_clusters=3, use_cols=None):
    if use_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return None, None, "数值列不足2列，无法聚类"
        use_cols = num_cols[:2]
    data = df[use_cols].dropna()
    if len(data) < n_clusters:
        return None, None, "数据点少于聚类数"
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='red', label='中心点')
    ax.set_xlabel(use_cols[0])
    ax.set_ylabel(use_cols[1])
    ax.set_title(f"KMeans 聚类 (k={n_clusters})")
    ax.legend()
    plt.colorbar(scatter)
    sil = silhouette_score(data_scaled, labels)
    return kmeans, fig, f"轮廓系数 = {sil:.3f}"

# ----------------------------- 自然语言生成图表 -----------------------------
def generate_chart_from_query(df, query):
    query_lower = query.lower()
    chart_type = "bar"
    if "柱状图" in query or "条形图" in query:
        chart_type = "bar"
    elif "折线图" in query or "趋势" in query:
        chart_type = "line"
    elif "饼图" in query or "占比" in query:
        chart_type = "pie"
    elif "分布" in query or "直方图" in query:
        chart_type = "hist"
    group_col = None
    value_col = None
    for col in df.columns:
        if col in query or ("按" + col) in query:
            group_col = col
            break
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in query:
            value_col = col
            break
    if not group_col or not value_col:
        return None, "无法自动识别分组列和数值列，请在问题中明确列名"
    if chart_type == "bar":
        grouped = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped.plot(kind='bar', ax=ax)
        ax.set_xlabel(group_col)
        ax.set_ylabel(value_col)
        ax.set_title(f"{value_col} 按 {group_col} 柱状图")
        plt.xticks(rotation=45, ha='right')
    elif chart_type == "line":
        if '日期' in df.columns or '时间' in df.columns:
            time_col = '日期' if '日期' in df.columns else '时间'
            df_sorted = df.sort_values(time_col)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_sorted[time_col], df_sorted[value_col], marker='o')
            ax.set_xlabel(time_col)
            ax.set_ylabel(value_col)
            ax.set_title(f"{value_col} 随时间变化")
            plt.xticks(rotation=45)
        else:
            grouped = df.groupby(group_col)[value_col].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            grouped.plot(kind='line', marker='o', ax=ax)
            ax.set_xlabel(group_col)
            ax.set_ylabel(value_col)
            ax.set_title(f"{value_col} 按 {group_col} 折线图")
    elif chart_type == "pie":
        grouped = df.groupby(group_col)[value_col].sum()
        fig, ax = plt.subplots(figsize=(8, 8))
        grouped.plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title(f"{value_col} 按 {group_col} 占比")
    elif chart_type == "hist":
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[value_col].dropna(), bins=20, edgecolor='black')
        ax.set_xlabel(value_col)
        ax.set_ylabel("频数")
        ax.set_title(f"{value_col} 分布直方图")
    else:
        return None, "不支持的图表类型"
    plt.tight_layout()
    return fig, f"成功生成{chart_type}图：{value_col} 按 {group_col}"

# ----------------------------- 质量报告函数 -----------------------------
def generate_quality_report(df, file_name):
    report = []
    report.append({"指标": "文件名", "值": file_name})
    report.append({"指标": "总行数", "值": len(df)})
    report.append({"指标": "总列数", "值": len(df.columns)})
    report.append({"指标": "重复行数", "值": df.duplicated().sum()})
    report.append({"指标": "内存占用(MB)", "值": round(df.memory_usage(deep=True).sum() / 1024**2, 2)})
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        missing_pct = round(missing / len(df) * 100, 2)
        unique = df[col].nunique()
        outliers = None
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
            outliers = f"{outlier_count} (阈值: [{lower:.2f}, {upper:.2f}])"
        sample = df[col].dropna().head(3).tolist()
        sample_str = ", ".join(str(s) for s in sample) if sample else "无"
        report.append({
            "指标": f"列: {col}",
            "值": f"类型: {dtype}, 缺失: {missing} ({missing_pct}%), 唯一值: {unique}, 异常值: {outliers if outliers else 'N/A'}, 示例: {sample_str}"
        })
    return pd.DataFrame(report)

# ----------------------------- 多表关联 -----------------------------
def merge_excel_files(df1, df2, join_key, how='inner'):
    df1 = df1.copy().fillna('')
    df2 = df2.copy().fillna('')
    if join_key not in df1.columns:
        return None, f"关联键 '{join_key}' 在第一个表中不存在。可用列：{', '.join(df1.columns)}"
    if join_key not in df2.columns:
        return None, f"关联键 '{join_key}' 在第二个表中不存在。可用列：{', '.join(df2.columns)}"
    merged = pd.merge(df1, df2, on=join_key, how=how)
    return merged, f"成功合并，共 {len(merged)} 行。"

# ----------------------------- session_state 初始化 -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_dfs" not in st.session_state:
    st.session_state.uploaded_dfs = {}
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "quality_report_df" not in st.session_state:
    st.session_state.quality_report_df = None
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "show_favorites" not in st.session_state:
    st.session_state.show_favorites = False
if "retriever_k" not in st.session_state:
    st.session_state.retriever_k = 4
if "use_history" not in st.session_state:
    st.session_state.use_history = True
if "last_result_df" not in st.session_state:
    st.session_state.last_result_df = None

# ----------------------------- 侧边栏 -----------------------------
with st.sidebar:
    st.header("📂 文档管理")
    uploaded_file = st.file_uploader("上传 Excel 文件", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.uploaded_dfs[uploaded_file.name] = df
            st.session_state.current_df = df
            st.session_state.current_filename = uploaded_file.name
            st.success(f"已加载：{uploaded_file.name}，共 {len(df)} 行")
            report_df = generate_quality_report(df, uploaded_file.name)
            st.session_state.quality_report_df = report_df
            with st.expander("📊 数据质量报告（点击展开）"):
                st.dataframe(report_df, use_container_width=True)
        except Exception as e:
            st.error(f"读取失败：{e}")
    
    st.markdown("---")
    if st.session_state.uploaded_dfs:
        st.subheader("已上传文件")
        for name in st.session_state.uploaded_dfs.keys():
            if st.button(f"📄 {name}", key=f"select_{name}"):
                st.session_state.current_df = st.session_state.uploaded_dfs[name]
                st.session_state.current_filename = name
                st.rerun()
    else:
        st.info("暂无文件，请上传")
    
    st.markdown("---")
    st.header("⚙️ 聊天设置")
    st.session_state.retriever_k = st.slider("检索数量 (k)", 1, 10, st.session_state.retriever_k)
    st.session_state.use_history = st.checkbox("对话历史记忆", value=st.session_state.use_history)
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.last_result_df = None
        st.rerun()
    
    st.markdown("---")
    st.header("📜 历史与收藏")
    if st.button("📋 显示查询历史"):
        st.session_state.show_history = not st.session_state.show_history
        st.session_state.show_favorites = False
        st.rerun()
    if st.button("⭐ 显示收藏夹"):
        st.session_state.show_favorites = not st.session_state.show_favorites
        st.session_state.show_history = False
        st.rerun()

# 显示历史记录
if st.session_state.show_history:
    with st.sidebar.expander("查询历史", expanded=True):
        data = load_history()
        for i, item in enumerate(data["history"]):
            st.markdown(f"**{item['timestamp']}**  \nQ: {item['query']}  \nA: {item['answer']}")
            if st.button(f"收藏", key=f"fav_{i}"):
                add_to_favorites(item['query'], item['answer'])
                st.success("已添加收藏")
        if st.button("清空历史"):
            save_history({"history": [], "favorites": data["favorites"]})
            st.rerun()

if st.session_state.show_favorites:
    with st.sidebar.expander("收藏夹", expanded=True):
        data = load_history()
        for idx, item in enumerate(data["favorites"]):
            st.markdown(f"**{item['timestamp']}**  \nQ: {item['query']}  \nA: {item['answer']}")
            if st.button(f"删除收藏", key=f"del_fav_{idx}"):
                remove_from_favorites(idx)
                st.rerun()

if st.session_state.quality_report_df is not None:
    with st.sidebar.expander("📋 已有质量报告（点击查看）"):
        st.dataframe(st.session_state.quality_report_df, use_container_width=True)

# ----------------------------- 导出辅助函数 -----------------------------
def export_result(df, filename_prefix="result"):
    if df is not None and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 导出结果 (CSV)",
            data=csv,
            file_name=f"{filename_prefix}_{int(time.time())}.csv",
            mime="text/csv",
            key=f"export_{int(time.time())}"
        )
    else:
        st.info("没有可导出的数据。")

# ----------------------------- 智能生成 Excel 报告 -----------------------------
def generate_excel_report(df, query):
    """根据用户描述生成统计报告（DataFrame）"""
    query_lower = query.lower()
    # 班级人员完整性校验
    if "班级" in query and ("完整性" in query or "人员" in query) or "每个班级的学生人数" in query:
        if '班级' not in df.columns:
            return None, "当前数据中没有'班级'列，无法统计。"
        count_df = df['班级'].value_counts().reset_index()
        count_df.columns = ['班级', '学生人数']
        count_df = count_df.sort_values('班级')
        return count_df, f"已生成班级学生人数统计表，共 {len(count_df)} 个班级。"
    # 专业分布
    elif "专业" in query and ("统计" in query or "人数" in query):
        if '专业' not in df.columns:
            return None, "当前数据中没有'专业'列。"
        major_df = df['专业'].value_counts().reset_index()
        major_df.columns = ['专业', '人数']
        return major_df, f"已生成专业人数统计表。"
    # 默认：返回数据概览（前100行）
    else:
        return df.head(100), "数据预览（前100行）。"

# ----------------------------- 聊天界面 -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入你的问题（智能识别统计、图表、脱敏、关联等）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    answer = ""
    if st.session_state.current_df is None:
        answer = "请先上传 Excel 文件。"
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        add_to_history(prompt, answer)
        st.session_state.last_result_df = None
    else:
        df = st.session_state.current_df.copy()
        # 1. 脱敏
        if "脱敏" in prompt:
            df_masked = mask_sensitive_columns(df)
            answer = f"已对全部敏感信息进行脱敏处理，共 {len(df_masked)} 行。预览：\n{df_masked.head(10).to_markdown()}"
            st.session_state.last_result_df = df_masked
            with st.chat_message("assistant"):
                st.markdown(answer)
                export_result(st.session_state.last_result_df, "脱敏后数据")
            st.session_state.messages.append({"role": "assistant", "content": answer})
            add_to_history(prompt, answer)
        # 2. 排序取前N
        elif re.search(r'前\s*(\d+)\s*名', prompt) and re.search(r'升序|降序|排序', prompt):
            match = re.search(r'前\s*(\d+)\s*名', prompt)
            if match:
                n = int(match.group(1))
                sort_col = "学号" if "学号" in df.columns else df.columns[0]
                df_sorted = df.sort_values(by=sort_col).head(n)
                answer = f"按{sort_col}升序前{n}条记录：\n{df_sorted.to_markdown()}"
                st.session_state.last_result_df = df_sorted
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    export_result(st.session_state.last_result_df, "排序结果")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
            else:
                answer = "请指定数字，例如「前10名」。"
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
                st.session_state.last_result_df = None
        # 3. 统计各班男女比例
        elif "统计" in prompt and "班级" in prompt and "男女" in prompt:
            if "班级" in df.columns and "性别" in df.columns:
                result = df.groupby("班级")["性别"].value_counts().unstack().fillna(0)
                answer = f"各班男女统计：\n{result.to_markdown()}"
                st.session_state.last_result_df = result.reset_index()
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    export_result(st.session_state.last_result_df, "男女统计")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
            else:
                answer = "缺少班级或性别列。"
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
                st.session_state.last_result_df = None
        # 4. 生成索引表
        elif "索引表" in prompt or "生成索引表" in prompt:
            needed_cols = []
            for col in ['序号', '姓名', '班级', '学号', '专业']:
                if col in df.columns:
                    needed_cols.append(col)
            if not needed_cols:
                answer = "数据中缺少必要的列，无法生成索引表。"
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
                st.session_state.last_result_df = None
            else:
                index_df = df[needed_cols].copy()
                if '序号' in index_df.columns:
                    index_df = index_df.sort_values('序号')
                answer = f"已生成索引表，包含 {len(index_df)} 条记录。预览：\n{index_df.head(10).to_markdown()}"
                st.session_state.last_result_df = index_df
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    export_result(st.session_state.last_result_df, "索引表")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
        # 5. 生成 Excel 报告（新增，普遍性）
        elif "生成" in prompt and ("excel" in prompt.lower() or "表格" in prompt or "报告" in prompt):
            report_df, msg = generate_excel_report(df, prompt)
            if report_df is not None:
                answer = msg + f"\n\n数据预览：\n{report_df.head(20).to_markdown()}"
                st.session_state.last_result_df = report_df
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    export_result(st.session_state.last_result_df, "报告")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
            else:
                answer = msg
                with st.chat_message("assistant"):
                    st.warning(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
                st.session_state.last_result_df = None
        # 6. 生成图表
        elif "图" in prompt and ("柱状图" in prompt or "折线图" in prompt or "饼图" in prompt or "直方图" in prompt):
            with st.spinner("📊 正在生成图表..."):
                fig, msg = generate_chart_from_query(df, prompt)
                if fig:
                    with st.chat_message("assistant"):
                        st.pyplot(fig)
                        st.caption(msg)
                    answer = msg
                else:
                    answer = msg
                    with st.chat_message("assistant"):
                        st.warning(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                add_to_history(prompt, answer)
                st.session_state.last_result_df = None
        # 7. 多表关联
        elif len(st.session_state.uploaded_dfs) >= 2 and ("关联" in prompt or "合并" in prompt or "连接" in prompt):
            files = list(st.session_state.uploaded_dfs.keys())
            with st.chat_message("assistant"):
                st.info("请选择要关联的两个表")
                col1, col2 = st.columns(2)
                with col1:
                    sel1 = st.selectbox("第一个表", files, key="merge1")
                with col2:
                    sel2 = st.selectbox("第二个表", files, key="merge2")
                join_key = st.text_input("关联键（列名）", placeholder="例如：学号", key="join_key_input")
                how = st.selectbox("连接类型", ["inner", "left", "right", "outer"], key="how_select")
                if st.button("执行关联"):
                    df1 = st.session_state.uploaded_dfs[sel1]
                    df2 = st.session_state.uploaded_dfs[sel2]
                    merged, msg = merge_excel_files(df1, df2, join_key, how)
                    if merged is not None:
                        st.markdown(f"### {msg}")
                        st.dataframe(merged)
                        st.session_state.last_result_df = merged
                        export_result(merged, "关联结果")
                        answer = msg
                    else:
                        st.error(msg)
                        answer = msg
                        st.session_state.last_result_df = None
                else:
                    answer = "等待用户选择"
                    st.session_state.last_result_df = None
            st.session_state.messages.append({"role": "assistant", "content": answer})
            add_to_history(prompt, answer)
        # 8. 相关性、回归、聚类
        elif "相关" in prompt or "回归" in prompt or "聚类" in prompt:
            # 这里只处理，不导出表格
            if "相关" in prompt and ("矩阵" in prompt or "热力图" in prompt):
                with st.spinner("📈 正在计算相关性..."):
                    corr, fig = compute_correlation(df)
                    if fig:
                        with st.chat_message("assistant"):
                            st.pyplot(fig)
                            st.dataframe(corr)
                        answer = "生成了相关系数热力图"
                    else:
                        answer = "无法计算相关性，缺少数值列"
                        with st.chat_message("assistant"):
                            st.warning(answer)
            elif "回归" in prompt:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) >= 2:
                    with st.spinner("📉 正在进行线性回归分析..."):
                        x_col = st.selectbox("选择自变量 (X)", num_cols, key="reg_x")
                        y_col = st.selectbox("选择因变量 (Y)", num_cols, key="reg_y")
                        if st.button("执行回归分析"):
                            model, fig, msg = linear_regression_analysis(df, x_col, y_col)
                            if fig:
                                st.pyplot(fig)
                                answer = msg
                            else:
                                answer = msg
                                st.error(answer)
                else:
                    answer = "数值列不足，无法进行回归分析"
                    st.warning(answer)
            elif "聚类" in prompt:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(num_cols) >= 2:
                    with st.spinner("🧬 正在进行 KMeans 聚类..."):
                        n_clusters = st.slider("聚类数量 (k)", 2, 10, 3)
                        use_cols = st.multiselect("选择用于聚类的列（默认前两个数值列）", num_cols, default=num_cols[:2])
                        if st.button("执行聚类"):
                            kmeans, fig, msg = kmeans_clustering(df, n_clusters, use_cols)
                            if fig:
                                st.pyplot(fig)
                                answer = msg
                            else:
                                answer = msg
                                st.error(answer)
                else:
                    answer = "数值列不足，无法进行聚类"
                    st.warning(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            add_to_history(prompt, answer)
            st.session_state.last_result_df = None
        # 9. 通用 LLM 问答
        else:
            with st.spinner("🤖 AI 正在思考中，请稍候..."):
                data_preview = df.head(20).to_markdown()
                col_stats = []
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        col_stats.append(f"- {col}: 最小值={df[col].min()}, 最大值={df[col].max()}, 均值={df[col].mean():.2f}, 非空={df[col].count()}/{len(df)}")
                    else:
                        unique_vals = df[col].dropna().unique()[:5]
                        col_stats.append(f"- {col}: 唯一值示例 {list(unique_vals)} (共{df[col].nunique()}种), 非空={df[col].count()}/{len(df)}")
                col_stats_text = "\n".join(col_stats)
                sys_prompt = f"""你是一个数据分析专家。请根据提供的Excel数据回答用户的问题。

数据总行数：{len(df)}
数据列：{list(df.columns)}

各列统计信息：
{col_stats_text}

数据预览（前20行）：
{data_preview}

用户问题：{prompt}

回答要求：
- 如果问题涉及计算，请基于全量数据计算并给出具体数字。
- 使用清晰的中文，必要时用列表或表格。
"""
                try:
                    resp = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "qwen2.5:7b", "prompt": sys_prompt, "stream": False},
                        timeout=120
                    )
                    answer = resp.json().get("response", "无响应")
                except Exception as e:
                    answer = f"连接 Ollama 失败：{e}\n请确保 Ollama 服务已启动（ollama serve）。"
            with st.chat_message("assistant"):
                st.markdown(answer)
                # 对于普通问答，不自动导出
            st.session_state.messages.append({"role": "assistant", "content": answer})
            add_to_history(prompt, answer)
            st.session_state.last_result_df = None