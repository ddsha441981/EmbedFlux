import requests
import logging
from typing import List, Optional, Dict
from langchain.schema import Document
import re
from urllib.parse import urlparse, urljoin
import time

logger = logging.getLogger(__name__)


class JinaWebProcessor:
    """Integration with Jina Reader API for web content processing"""

    def __init__(self):
        self.base_url = "https://r.jina.ai/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Programming-Query-System/1.0'
        })
        self.rate_limit_delay = 1  # seconds between requests
        self.last_request_time = 0

    def _respect_rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch clean content from URL using Jina Reader"""
        try:
            self._respect_rate_limit()

            # Use Jina Reader API
            response = self.session.get(f"{self.base_url}{url}", timeout=30)
            response.raise_for_status()

            content = response.text

            # Basic content validation
            if len(content) < 100:  # Too short, probably an error
                logger.warning(f"Content too short for {url}")
                return None

            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def process_programming_urls(self, urls: List[str]) -> List[Document]:
        """Process multiple URLs and convert to documents"""
        documents = []

        for url in urls:
            try:
                content = self.fetch_url_content(url)
                if content:
                    # Create document with metadata
                    doc = Document(
                        page_content=content[:5000],  # Limit content size
                        metadata={
                            'source': url,
                            'content_type': 'web',
                            'processed_by': 'jina_reader',
                            'programming_related': self._is_programming_content(content),
                            'domain': self._extract_domain(url),
                            'processed_at': time.time()
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Successfully processed: {url}")

            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue

        return documents

    def _is_programming_content(self, content: str) -> bool:
        """Detect if content is programming-related"""
        programming_indicators = [
            'function', 'class', 'method', 'variable', 'algorithm',
            'code', 'programming', 'syntax', 'documentation',
            'library', 'framework', 'api', 'developer', 'software',
            'import', 'export', 'const', 'var', 'def', 'public',
            'private', 'static', 'void', 'return', 'if', 'else',
            'for', 'while', 'try', 'catch', 'exception'
        ]

        content_lower = content.lower()
        matches = sum(1 for indicator in programming_indicators
                      if indicator in content_lower)

        # Consider it programming content if it has at least 5 indicators
        return matches >= 5

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    def search_stackoverflow(self, query: str, max_results: int = 3) -> List[Document]:
        """Specific method for Stack Overflow searches"""
        try:
            # This is a simplified implementation
            # In practice, you might want to use Stack Overflow's API
            search_url = f"https://stackoverflow.com/search?q={query.replace(' ', '+')}"

            content = self.fetch_url_content(search_url)
            if content:
                return [Document(
                    page_content=content,
                    metadata={
                        'source': search_url,
                        'content_type': 'stackoverflow',
                        'query': query,
                        'programming_related': True
                    }
                )]
        except Exception as e:
            logger.error(f"Stack Overflow search failed: {e}")

        return []

    def get_documentation(self, language: str) -> List[str]:
        """Get documentation URLs for specific programming languages"""
        doc_urls = {
            'python': [
                'https://docs.python.org/3/tutorial/',
                'https://docs.python.org/3/library/',
            ],
            'javascript': [
                'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide',
                'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference',
            ],
            'java': [
                'https://docs.oracle.com/en/java/javase/17/docs/api/',
                'https://docs.oracle.com/javase/tutorial/',
            ],
            'react': [
                'https://react.dev/learn',
                'https://react.dev/reference/react',
            ]
        }

        return doc_urls.get(language.lower(), [])
