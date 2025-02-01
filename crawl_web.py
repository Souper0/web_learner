import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
import psutil
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
import sqlite3
from bs4 import BeautifulSoup
from defusedxml.ElementTree import parse  # 更安全的XML解析
from io import BytesIO

load_dotenv()

llm_chat_client = AsyncOpenAI(
    api_key=os.getenv("CHAT_API_KEY"),
    base_url=os.getenv("CHAT_BASE_URL")  
)

llm_embedding_client = AsyncOpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL")  
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # 优先在代码块边界分割
        code_block = text[start:end].rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block + 3
        
        # 其次在段落边界分割
        elif '\n\n' in text[start:end]:
            last_break = text[start:end].rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        
        # 最后在句子边界分割
        elif '. ' in text[start:end]:
            last_period = text[start:end].rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:

    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await llm_chat_client.chat.completions.create(
            model=os.getenv("CHAT_MODEL", "qwen-plus"),  
            messages=[
                {"role": "system", "content": system_prompt},

                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={ "type": "json_object" }
        )
        print(response.choices[0].message.content)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding from QWen instead of OpenAI"""
    try:
        response = await llm_embedding_client.embeddings.create(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),  
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1024  #! note: adjust based on the embedding model

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": os.getenv("SOURCE_NAME"),
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    conn = sqlite3.connect('local_docs.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO site_pages 
            (url, chunk_number, title, summary, content, metadata, embedding, created_at)
            VALUES (?,?,?,?,?,?,?,?)
        ''', (  # 明确指定列名
            chunk.url,
            chunk.chunk_number,
            chunk.title,
            chunk.summary,
            chunk.content,
            json.dumps(chunk.metadata),
            json.dumps(chunk.embedding),
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
    except Exception as e:
        print(f"插入失败: {e}")
    finally:
        conn.close()

@dataclass
class ContentFilter:
    include_keywords: List[str] = None
    exclude_keywords: List[str] = None
    css_selectors: List[str] = None  # 例如 ["article.main-content", "div.doc-section"]
    max_content_length: int = 100000  # 防止抓取过大页面

def should_keep_content(content: str, filter: ContentFilter) -> bool:
    """内容过滤逻辑"""
    # 长度过滤
    if len(content) > filter.max_content_length:
        return False
    
    # 关键词过滤
    content_lower = content.lower()
    if filter.include_keywords:
        if not any(kw in content_lower for kw in filter.include_keywords):
            return False
    if filter.exclude_keywords:
        if any(kw in content_lower for kw in filter.exclude_keywords):
            return False
    return True

async def process_and_store_document(url: str, markdown: str, filter: ContentFilter = None):
    """新增内容过滤参数"""
    if filter and not should_keep_content(markdown, filter):
        print(f"跳过不符合条件的内容: {url}")
        return
    
    # 原有处理逻辑...
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def discover_links(base_url: str, html: str) -> List[str]:
    """基于目录结构的优先级链接发现"""
    soup = BeautifulSoup(html, 'html.parser')
    priority_links = []
    other_links = []
    
    # 常见目录结构识别模式
    directory_selectors = [
        ('nav', {}),
        ('div[class*="sidebar"]', {}),
        ('div[class*="toc"]', {}),
        ('div[class*="menu"]', {}),
        ('ul[class*="nav"]', {}),
        ('div[role="navigation"]', {}),
        ('a[href*="/chapter"]', {}),
        ('a[href*="/section"]', {}),
    ]
    
    for selector, attrs in directory_selectors:
        for element in soup.select(selector, attrs):
            for a in element.find_all('a', href=True):
                full_url = urljoin(base_url, a['href'])
                if full_url not in priority_links:
                    priority_links.append(full_url)
    
    # 提取其他链接并去重
    for a in soup.find_all('a', href=True):
        full_url = urljoin(base_url, a['href'])
        if full_url not in priority_links and full_url not in other_links:
            if is_valid_link(full_url):
                other_links.append(full_url)
    
    return priority_links + other_links

def is_valid_link(url: str) -> bool:
    """过滤无效链接"""
    parsed = urlparse(url)
    return not any(ext in parsed.path for ext in ['.pdf', '.png', '.jpg']) 

async def unified_crawler(
    base_url: str,
    max_depth: int = 3,
    max_concurrent: int = 8,
    max_pages: int = None,
    time_limit: int = None
) -> List[str]:
    """整合后的统一抓取入口"""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    
    # 初始化配置
    crawler = AsyncWebCrawler(config=BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--no-sandbox"]
    ))
    await crawler.start()
    
    visited = set()
    queue = [(base_url, 0)]  # (url, depth)
    results = []
    start_time = datetime.now()
    
    async def process_page(url: str, depth: int):
        """统一处理页面"""
        if url in visited or depth > max_depth:
            return []
        visited.add(url)
        
        try:
            # 执行抓取
            result = await crawler.arun(url, CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
            ))
            
            if not result.success:
                return []
            
            # 处理内容
            await process_and_store_document(url, result.markdown_v2.raw_markdown)
            results.append(url)
            
            # 发现新链接
            new_links = await discover_links(base_url, result.html)
            return [(link, depth+1) for link in new_links if link not in visited]
        
        except Exception as e:
            print(f"处理 {url} 失败: {str(e)[:100]}")
            return []
    
    try:
        while queue and len(results) < (max_pages or float('inf')):
            # 检查时间限制
            if time_limit and (datetime.now() - start_time).seconds > time_limit:
                print(f"达到时间限制 {time_limit}秒")
                break
            
            # 批量处理
            batch = queue[:max_concurrent]
            del queue[:max_concurrent]
            
            tasks = [process_page(url, depth) for url, depth in batch]
            new_links = await asyncio.gather(*tasks)
            
            # 合并新链接到队列
            for links in new_links:
                queue.extend(links)
            
            # 进度报告
            if len(results) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"已抓取 {len(results)} 页 | 耗时: {elapsed:.1f}s")
    
    finally:
        await crawler.close()
    
    return results

def is_directory_link(url: str) -> bool:
    """基于URL模式识别目录链接"""
    patterns = [
        r'/chapter/', r'/section\d+', r'/toc/', 
        r'/nav/', r'index\.html$', r'_sidebar\.md'
    ]
    return any(re.search(p, url) for p in patterns)

def inspect_database(limit=3):
    """快速查看数据库内容"""
    conn = sqlite3.connect('local_docs.db')
    cursor = conn.cursor()
    
    # 获取总条目数
    cursor.execute("SELECT COUNT(*) FROM site_pages")
    total = cursor.fetchone()[0]
    print(f"Total chunks: {total}\n")
    
    # 获取示例数据
    cursor.execute("SELECT * FROM site_pages LIMIT ?", (limit,))
    rows = cursor.fetchall()
    
    # 打印列名
    col_names = [description[0] for description in cursor.description]
    print("|".join(col_names))
    print("-"*80)
    
    # 打印数据
    for row in rows:
        shortened = [str(x)[:50]+"..." if len(str(x))>50 else x for x in row]
        print("|".join(map(str, shortened)))
    
    conn.close()

def init_db():
    conn = sqlite3.connect('local_docs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS site_pages (
            url TEXT,
            chunk_number INTEGER,
            title TEXT,
            summary TEXT,
            content TEXT,
            metadata TEXT,
            embedding TEXT,
            created_at DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def init_db_required(func):
    def wrapper(*args, **kwargs):
        init_db()
        return func(*args, **kwargs)
    return wrapper


# 添加内存监控
process = psutil.Process()
def report_resources():
    mem = process.memory_info().rss / 1024 / 1024
    print(f"内存使用: {mem:.1f}MB")

# 添加更详细的进度报告
def report_progress(processed: int, total: int, start_time: datetime):
    elapsed = (datetime.now() - start_time).total_seconds()
    pages_per_sec = processed / elapsed if elapsed > 0 else 0
    remaining = (total - processed) / pages_per_sec if pages_per_sec > 0 else 0
    
    print(f"\n进度: {processed}/{total} ({processed/total:.1%})")
    print(f"已用时间: {elapsed:.1f}s")
    print(f"预估剩余时间: {remaining:.1f}s")
    print(f"速度: {pages_per_sec:.1f} 页/秒")

async def main():
    """修复后的主函数交互流程"""
    base_url = os.getenv("WEB_URL")
    if not base_url:
        print("错误：未设置WEB_URL环境变量")
        return
    
    print(f"\n=== 抓取配置 ===")
    print(f"目标网站: {base_url}")
    
    # 用户设置限制
    max_pages = None
    if input("\n是否设置最大抓取页数？(y/n) ").lower() == 'y':
        while True:
            try:
                max_pages = int(input("请输入最大页数: "))
                break
            except ValueError:
                print("输入无效，请重新输入数字！")
    
    time_limit = None
    if input("\n是否设置时间限制（秒）？(y/n) ").lower() == 'y':
        while True:
            try:
                time_limit = int(input("请输入时间限制: "))
                break
            except ValueError:
                print("输入无效，请重新输入数字！")
    
    # 显示最终配置
    print("\n=== 开始抓取 ===")
    print(f"并发数: 8")
    print(f"最大页数: {max_pages or '无限制'}")
    print(f"时间限制: {time_limit or '无限制'}秒")
    
    # 获取URL并抓取
    urls = await unified_crawler(
        base_url,
        max_depth=3,
        max_concurrent=8,
        max_pages=max_pages,
        time_limit=time_limit
    )
    init_db()
    
    print("开始抓取...")
    await unified_crawler(
        base_url,
        max_depth=3,
        max_concurrent=8,
        max_pages=max_pages,
        time_limit=time_limit
    )
    inspect_database()


if __name__ == "__main__":
    asyncio.run(main())
