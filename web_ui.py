import gradio as gr
import sqlite3
import json
import asyncio
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from crawl_web import crawl_parallel, get_urls_from_sitemap
from rag_cli import retrieve_chunks, generate_answer
import os

def get_db_connection():
    """为每个线程创建独立的数据库连接"""
    return sqlite3.connect('local_docs.db', check_same_thread=False)

def build_knowledge_graph():
    """修改后的知识图谱生成函数"""
    conn = get_db_connection()
    cursor = conn.cursor()
    G = nx.DiGraph()
    
    # 获取所有页面
    cursor.execute("SELECT url, metadata FROM site_pages")
    pages = {}
    for url, metadata in cursor.fetchall():
        meta = json.loads(metadata)
        if url not in pages:
            G.add_node(url, 
                      title=meta.get('url_path', ''),
                      size=meta.get('chunk_size', 0))
            pages[url] = []
        pages[url].append(meta)
    
    # 添加关联关系
    for url in pages:
        cursor.execute("SELECT content FROM site_pages WHERE url=?", (url,))
        content = " ".join([row[0] for row in cursor.fetchall()])
        if "http" in content:
            links = [link for link in pages.keys() if link in content]
            for link in links:
                G.add_edge(url, link)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=200, font_size=8)
    conn.close()  # 添加关闭连接
    return plt.gcf()

async def rag_pipeline(question: str):
    """修改后的RAG流程"""
    conn = get_db_connection()
    cursor = conn.cursor()
    from rag_cli import get_embedding  # 延迟导入避免冲突
    
    question_embedding = await get_embedding(question)
    chunks = retrieve_chunks(question_embedding)
    
    if not chunks:
        return "没有找到相关上下文信息", ""
    
    context = "\n\n".join([f"[来源：{c['metadata']['url_path']}]\n{c['content']}" for c in chunks])
    answer = await generate_answer(question, context)
    sources = "\n".join([f"- {c['source']}" for c in chunks])
    conn.close()
    return answer, sources

def get_db_data():
    """线程安全的数据库查询"""
    conn = sqlite3.connect('local_docs.db')
    cursor = conn.cursor()
    cursor.execute("SELECT url, COUNT(*) FROM site_pages GROUP BY url")
    data = cursor.fetchall()
    conn.close()
    return data

def get_db_stats():
    """获取统计信息"""
    conn = sqlite3.connect('local_docs.db')
    cursor = conn.cursor()
    stats = {
        "总页面数": cursor.execute("SELECT COUNT(DISTINCT url) FROM site_pages").fetchone()[0],
        "总分块数": cursor.execute("SELECT COUNT(*) FROM site_pages").fetchone()[0],
        "存储大小": f"{os.path.getsize('local_docs.db')/1024/1024:.2f} MB"
    }
    conn.close()
    return stats

def build_ui():
    """构建完整的前端界面"""
    with gr.Blocks(title="知识库管理系统") as demo:
        gr.Markdown("# 🧠 智能知识库管理系统")
        
        with gr.Tab("🌐 网页抓取"):
            with gr.Row():
                url_input = gr.Textbox(label="输入网址", placeholder="https://example.com")
                sitemap_btn = gr.Button("从sitemap抓取")
            crawl_btn = gr.Button("开始抓取", variant="primary")
            progress = gr.Slider(visible=False, label="抓取进度")
            log_output = gr.Textbox(label="操作日志", interactive=False)
        
        with gr.Tab("❓ 问答"):
            with gr.Row():
                question_input = gr.Textbox(label="输入问题", lines=3)
                answer_output = gr.Markdown(label="答案")
            with gr.Accordion("查看参考来源", open=False):
                sources_output = gr.Markdown()
            ask_btn = gr.Button("提交问题", variant="primary")
        
        with gr.Tab("📊 知识图谱"):
            plot = gr.Plot(label="知识关联图谱", every=60)
            gr.Button("刷新图谱").click(build_knowledge_graph, outputs=plot)
        
        with gr.Tab("📂 知识库管理"):
            with gr.Row():
                stats = gr.JSON(label="统计信息")
                with gr.Column():
                    db_view = gr.Dataframe(
                        headers=["URL", "分块数"],
                        datatype=["str", "number"],
                        interactive=False
                    )
                    gr.Button("刷新数据").click(
                        get_db_data,  # 直接使用封装好的函数
                        outputs=db_view
                    )
            gr.Button("更新统计").click(
                get_db_stats,  # 使用封装好的统计函数
                outputs=stats
            )

        # 事件处理
        crawl_btn.click(
            fn=lambda url: asyncio.run(crawl_parallel([url])),
            inputs=url_input,
            outputs=log_output
        )
        
        sitemap_btn.click(
            lambda: (get_urls_from_sitemap(), "开始抓取sitemap..."),
            outputs=[url_input, log_output]
        )
        
        ask_btn.click(
            rag_pipeline,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        demo.load(build_knowledge_graph, outputs=plot)

    return demo

if __name__ == "__main__":
    # 初始化知识图谱
    build_knowledge_graph()
    
    # 启动前端
    web_ui = build_ui()
    web_ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 