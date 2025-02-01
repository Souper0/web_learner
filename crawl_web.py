import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

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

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

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
    """从HTML中提取站内链接"""
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == urlparse(base_url).netloc:
            links.append(full_url)
    return list(set(links))  # 去重

async def crawl_without_sitemap(base_url: str, max_depth: int = 2) -> List[str]:
    """无sitemap时的递归抓取"""
    browser_config = BrowserConfig(headless=True)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    visited = set()
    to_crawl = [(base_url, 0)]
    all_urls = []
    
    try:
        while to_crawl:
            url, depth = to_crawl.pop(0)
            if depth > max_depth or url in visited:
                continue
                
            result = await crawler.arun(url, config=CrawlerRunConfig())
            if result.success:
                all_urls.append(url)
                # 提取新链接
                new_links = await discover_links(base_url, result.html)
                to_crawl += [(link, depth+1) for link in new_links]
                
            visited.add(url)
    finally:
        await crawler.close()
    
    return all_urls

def get_urls_from_sitemap() -> List[str]:
    """改进的sitemap解析方法"""
    if not os.getenv("WEB_URL"):
        print("warning: WEB_URL is not set")
        return []
    
    sitemap_url = os.getenv("WEB_URL") + "/sitemap.xml"
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        # 使用defusedxml防止XML攻击
        tree = parse(BytesIO(response.content))
        root = tree.getroot()
        
        # 处理可能的sitemap索引文件
        if root.tag.endswith('sitemapindex'):
            sitemaps = [loc.text for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
            urls = []
            for sitemap in sitemaps:
                try:
                    sub_response = requests.get(sitemap, timeout=5)
                    sub_tree = parse(BytesIO(sub_response.content))
                    urls += [loc.text for loc in sub_tree.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
                except Exception as e:
                    print(f"解析子sitemap失败: {sitemap} - {e}")
            return urls
        
        # 直接解析普通sitemap
        return [loc.text for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
    
    except Exception as e:
        print(f"使用sitemap失败: {e}, 尝试递归抓取")
        base_url = os.getenv("WEB_URL")
        if not base_url:
            return []
        return asyncio.run(crawl_without_sitemap(base_url))

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

async def main():
    # Get URLs from sitemap.xml
    urls = get_urls_from_sitemap()
    if not urls:
        print("warning: No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    init_db_required(init_db)()
    await crawl_parallel(urls)
    inspect_database()

if __name__ == "__main__":
    asyncio.run(main())
