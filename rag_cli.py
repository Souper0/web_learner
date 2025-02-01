import sqlite3
import json
import os
import asyncio
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# 复用已有的OpenAI客户端配置
llm_chat_client = AsyncOpenAI(
    api_key=os.getenv("CHAT_API_KEY"),
    base_url=os.getenv("CHAT_BASE_URL")  
)

llm_embedding_client = AsyncOpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL")  
)

async def get_embedding(text: str) -> List[float]:
    """复用已有的embedding获取函数"""
    try:
        response = await llm_embedding_client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),  
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1024

def get_rag_db_connection():
    """创建线程安全的数据库连接"""
    return sqlite3.connect('local_docs.db', check_same_thread=False)

def retrieve_chunks(question_embedding: List[float], top_k: int = 3) -> List:
    """优化后的检索实现"""
    conn = get_rag_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT url, chunk_number, content, metadata, embedding FROM site_pages')
        chunks = []
        for row in cursor.fetchall():
            emb = json.loads(row[4])  # embedding在第五列
            similarity = cosine_similarity([question_embedding], [emb])[0][0]
            chunks.append({
                "similarity": similarity,
                "content": row[2],
                "metadata": json.loads(row[3]),
                "source": row[0]
            })
        
        # 按相似度排序并返回前k个
        return sorted(chunks, key=lambda x: x["similarity"], reverse=True)[:top_k]
    except Exception as e:
        print(f"检索失败: {e}")
        return []
    finally:
        conn.close()  # 确保关闭连接

async def generate_answer(question: str, context: str) -> str:
    """优化后的答案生成"""
    system_prompt = """基于以下上下文回答问题。请遵循：
    1. 如果上下文不足，请明确说明
    2. 保持答案简洁专业
    3. 使用Markdown格式
    
    上下文：
    {context}"""
    
    try:
        response = await llm_chat_client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "deepseek-reasoner"),
            messages=[
                {"role": "system", "content": system_prompt.format(context=context)},
                {"role": "user", "content": question}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成答案时出错: {str(e)}"

async def rag_process(question: str):
    """完整的RAG处理流程"""
    # 获取问题embedding
    question_embedding = await get_embedding(question)
    
    # 检索相关段落
    chunks = retrieve_chunks(question_embedding, top_k=7)
    
    if not chunks:
        print("没有找到相关上下文信息")
        return
    
    # 组合上下文
    context = "\n\n".join([
        f"[来源：{c['metadata']['url_path']} (相似度: {c['similarity']:.5f})]\n{c['content']}" 
        for c in chunks
    ])
    
    # 生成答案
    answer = await generate_answer(question, context)
    print("\n答案：")
    print("="*50)
    print(answer)
    print("="*50)
    print("\n相关来源：")
    for c in chunks:
        print(f"- {c['source']} (相似度: {c['similarity']:.4f})")

def main():
    """命令行交互主循环"""
    print("""
    命令行问答系统（输入exit退出）
    ================================
    """)
    
    while True:
        question = input("\n请输入问题： ").strip()
        if question.lower() in ["exit", "quit"]:
            break
            
        if not question:
            continue
            
        # 运行异步任务
        asyncio.run(rag_process(question))

if __name__ == "__main__":
    main()
