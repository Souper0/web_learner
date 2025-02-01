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

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

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

async def crawl_without_sitemap(base_url: str, max_depth: int = 2, max_concurrent: int = 8) -> List[str]:
    """基于目录优先的并行递归抓取"""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    
    crawler = AsyncWebCrawler(config=BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--no-sandbox"]
    ))
    await crawler.start()
    
    visited = set()
    priority_queue = []
    normal_queue = [(base_url, 0)]
    all_urls = []
    
    async def process_batch(batch: list) -> list:
        tasks = []
        for url, depth in batch:
            if url in visited or depth > max_depth:
                continue
            visited.add(url)
            tasks.append(crawler.arun(url, CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
            )))
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    try:
        while normal_queue or priority_queue:
            # 优先处理目录链接
            current_batch = []
            while len(current_batch) < max_concurrent and priority_queue:
                current_batch.append(priority_queue.pop(0))
            
            # 补充普通链接
            while len(current_batch) < max_concurrent and normal_queue:
                current_batch.append(normal_queue.pop(0))
            
            results = await process_batch(current_batch)
            
            for (url, depth), result in zip(current_batch, results):
                if isinstance(result, Exception):
                    continue
                
                all_urls.append(url)
                new_links = await discover_links(base_url, result.html)
                
                for link in new_links:
                    if link not in visited:
                        if is_directory_link(link):
                            priority_queue.append((link, depth+1))
                        else:
                            normal_queue.append((link, depth+1))
            
            # 内存优化
            if len(all_urls) % 10 == 0:
                await crawler._cleanup_sessions()
                
    finally:
        await crawler.close()
    
    return all_urls

def is_directory_link(url: str) -> bool:
    """基于URL模式识别目录链接"""
    patterns = [
        r'/chapter/', r'/section\d+', r'/toc/', 
        r'/nav/', r'index\.html$', r'_sidebar\.md'
    ]
    return any(re.search(p, url) for p in patterns)

async def get_crawl_urls(base_url: str) -> List[str]:
    """获取待抓取URL列表（优先sitemap）"""
    if urls := get_urls_from_sitemap():
        return urls
    print("启用递归抓取...")
    return await crawl_without_sitemap(base_url)

def get_urls_from_sitemap() -> List[str]:
    """健壮的sitemap解析实现"""
    try:
        sitemap_url = f"{os.getenv('WEB_URL')}/sitemap.xml"
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        tree = parse(BytesIO(response.content))
        root = tree.getroot()
        
        # 处理嵌套sitemap
        if root.tag.endswith('sitemapindex'):
            sitemaps = [loc.text for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
            urls = []
            for sitemap in sitemaps[:3]:  # 限制嵌套深度
                try:
                    sub_resp = requests.get(sitemap, timeout=5)
                    sub_tree = parse(BytesIO(sub_resp.content))
                    urls += [loc.text for loc in sub_tree.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
                except Exception:
                    continue
            return urls
        
        return [loc.text for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
    
    except Exception as e:
        print(f"Sitemap解析失败: {str(e)[:200]}")
        return []

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

async def estimate_total_pages(base_url: str) -> Tuple[int, str]:
    """多策略页面估算"""
    if sitemap_urls := get_urls_from_sitemap():
        return (len(sitemap_urls), "sitemap")
    
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(f"{base_url}/tutorials")
            if result.success:
                soup = BeautifulSoup(result.html, 'html.parser')
                pagination = soup.select('.pagination a')
                if pagination:
                    max_page = max(int(p.text) for p in pagination if p.text.isdigit())
                    return (max_page * 20, "pagination")  # 假设每页20项
    except:
        pass
    
    return (await crawl_without_sitemap(base_url, max_depth=1), "heuristic")

async def main():
    base_url = os.getenv("WEB_URL")
    if not base_url:
        print("错误：未设置WEB_URL环境变量")
        return
    
    # 预估页面数
    total, method = await estimate_total_pages(base_url)
    
    if input(f"是否继续抓取？(y/n) ").lower() != 'y':
        return
    
    # 获取并处理URL
    urls = await get_crawl_urls(base_url)
    init_db_required(init_db)()
    
    # 并行处理
    await crawl_parallel(urls, max_concurrent=8)
    inspect_database()

if __name__ == "__main__":
    asyncio.run(main())
