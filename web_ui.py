import gradio as gr
import sqlite3
import json
import asyncio
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from crawl_web import init_db, init_db_required, unified_crawler
from rag_cli import retrieve_chunks, generate_answer
import os

def get_db_connection():
    """为每个线程创建独立的数据库连接"""
    return sqlite3.connect('local_docs.db', check_same_thread=False)

def build_knowledge_graph():
    """添加初始化装饰器"""
    @init_db_required
    def _build_graph():
        conn = get_db_connection()
        cursor = conn.cursor()
        G = nx.DiGraph()
        
        # 获取所有页面并处理标题
        cursor.execute("SELECT url, metadata, COUNT(*) as chunk_count FROM site_pages GROUP BY url")
        pages = {}
        for url, metadata, chunk_count in cursor.fetchall():
            meta = json.loads(metadata)
            # 修改标题生成逻辑
            if 'url_path' in meta and meta['url_path']:
                # 使用 URL 路径的最后一部分作为标题
                title = meta['url_path'].rstrip('/').split('/')[-1]
            else:
                # 如果没有 url_path，则使用 URL 的最后一部分
                title = url.rstrip('/').split('/')[-1]
            
            # 如果标题为空，使用域名
            if not title:
                title = url.split('/')[2]  # 获取域名部分
            
            # 美化标题
            title = title.replace('-', ' ').replace('_', ' ').title()
            # 限制标题长度，确保可读性
            title = title[:30] + '...' if len(title) > 30 else title
            
            pages[url] = {
                "title": title,
                "size": chunk_count * 50 + 100,
                "depth": len(meta.get('url_path', '').split('/'))
            }
            G.add_node(url, **pages[url])
        
        # 如果没有数据时显示空图
        if len(pages) == 0:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
            plt.axis('off')
            conn.close()
            return plt.gcf()
        
        # 添加关联关系
        cursor.execute("SELECT url, content FROM site_pages")
        content_map = {url: content for url, content in cursor.fetchall()}
        
        for url in pages:
            if url in content_map and "http" in content_map[url]:
                links = [link for link in pages.keys() if link in content_map[url]]
                for link in links:
                    G.add_edge(url, link)
        
        plt.figure(figsize=(12, 8))
        pos = nx.kamada_kawai_layout(G)  # 使用更清晰的布局算法
        
        # 根据路径深度设置颜色
        colors = [node[1]['depth'] for node in G.nodes(data=True)]
        # 根据分块数量设置大小
        sizes = [node[1]['size'] for node in G.nodes(data=True)]
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_color=colors,
            cmap=plt.cm.viridis,
            node_size=sizes,
            alpha=0.9,
            edgecolors='grey'
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, pos,
            width=1,
            alpha=0.3,
            edge_color='gray',
            arrowsize=10
        )
        
        # 修改标签绘制，确保标签可见
        labels = {n[0]: n[1]['title'] for n in G.nodes(data=True)}
        print("labels: ",labels)
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,  # 减小字体以避免重叠
            font_color='black',
            font_weight='bold',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.7,
                pad=0.5
            )
        )
        
        # 添加颜色条和图例（添加有效性检查）
        if colors and len(set(colors)) > 1:  # 只在颜色数据有效时添加
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis,
                norm=plt.Normalize(vmin=min(colors), vmax=max(colors))
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)  # 显式指定ax参数
            cbar.set_label('URL Path Depth')
        elif colors:  # 所有节点颜色相同的情况
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis,
                norm=plt.Normalize(vmin=0, vmax=1)  # 设置默认范围
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.8)
            cbar.set_label('URL Path Depth (All Same)')
        
        plt.title("Knowledge Graph", fontsize=16)
        plt.axis('off')
        # nx.draw(G, pos, with_labels=True)
        conn.close()
        return plt.gcf()
    return _build_graph()

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
            with gr.Row():
                max_pages_input = gr.Number(label="最大抓取页数", value=100, precision=0)
                time_limit_input = gr.Number(label="时间限制(秒)", value=300, precision=0)
            with gr.Row():
                crawl_btn = gr.Button("开始抓取", variant="primary")
                # stop_btn = gr.Button("停止抓取")
            progress = gr.Slider(visible=True, label="抓取进度", interactive=False)
            gr.Markdown("### 实时统计")
            stats_panel = gr.JSON(label="抓取统计", value={
                "已抓取页面": 0,
                "剩余页面": 0,
                "预计剩余时间": "N/A"
            }, every=5)
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
            fn=lambda url, max_p, time_l: asyncio.run(unified_crawler(base_url=url, max_pages=max_p, time_limit=time_l)),
            inputs=[url_input, max_pages_input, time_limit_input],
            outputs=[stats_panel, log_output]
        )
        
        
        ask_btn.click(
            rag_pipeline,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        # stop_btn.click(
        #     fn=lambda: setattr(crawler, 'should_stop', True),
        #     outputs=None
        # )
        
        demo.load(build_knowledge_graph, outputs=plot)

    return demo

if __name__ == "__main__":
    init_db()  # 显式初始化
    # 初始化知识图谱
    build_knowledge_graph()
    
    # 启动前端
    web_ui = build_ui()
    web_ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

async def start_crawling(url: str, max_pages: int, time_limit: int):
    from crawl_web import unified_crawler
    try:
        await unified_crawler(
            base_url=url,
            max_depth=3,
            max_concurrent=8,
            max_pages=max_pages,
            time_limit=time_limit
        )
        return get_db_stats(), "抓取完成！"
    except Exception as e:
        return get_db_stats(), f"抓取中断: {str(e)}"

def get_crawl_stats():
    conn = sqlite3.connect('local_docs.db')
    cursor = conn.cursor()
    stats = {
        "已抓取页面": cursor.execute("SELECT COUNT(DISTINCT url) FROM site_pages").fetchone()[0],
        "剩余页面": "N/A",  # 需要crawler提供实时数据
        "预计剩余时间": "N/A"  # 需要计算逻辑
    }
    conn.close()
    return stats 