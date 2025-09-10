import os
import json
import hashlib
import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass
import numpy as np
import re
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_milvus import Milvus
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.schema import Document
from langchain_groq import ChatGroq
from mem0 import Memory

# Import our new modules
from web_content_processor import JinaWebProcessor
from enhanced_memory import EnhancedProgrammingMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# NEW: Enhanced embedding models configuration
PROGRAMMING_EMBEDDING_MODELS = {
    'fast-general': 'sentence-transformers/all-MiniLM-L12-v2',
    'code-focused': 'microsoft/codebert-base',
    'multilingual-code': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'current-default': 'sentence-transformers/all-mpnet-base-v2',
    'programming-optimized': 'sentence-transformers/all-MiniLM-L12-v2'  # Best balance
}


class ProgrammingLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    TYPESCRIPT = "typescript"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    BASH = "bash"
    POWERSHELL = "powershell"
    GENERAL = "general"


@dataclass
class ProgrammingQueryConfig:
    """Enhanced configuration for programming-specific queries"""
    top_k_per_strategy: int = 10
    final_top_k: int = 15
    score_threshold: float = 0.6
    mmr_diversity: float = 0.7
    enable_code_context: bool = True
    enable_cross_language: bool = True
    include_syntax_examples: bool = True
    enable_caching: bool = True
    combine_all_strategies: bool = True

    # NEW: Enhanced features
    enable_web_search: bool = True  # Web integration toggle
    embedding_model_type: str = 'programming-optimized'  # Model selection
    enhanced_memory: bool = True  # Advanced memory features
    max_web_results: int = 3  # Limit web results


@dataclass
class ProgrammingQueryResult:
    """Enhanced result for programming queries"""
    documents: List[Document]
    synthesized_answer: str
    scores: List[float]
    detected_languages: List[str]
    code_snippets: List[str]
    concepts: List[str]
    related_topics: List[str]
    query_strategies_used: List[str]
    query_time: float
    total_results: int
    confidence_score: float

    # NEW: Enhanced result fields
    web_sources_count: int = 0
    memory_context_used: bool = False
    embedding_model_used: str = ""
    personalization_strength: float = 0.0


class ProgrammingQueryEngine:
    """Enhanced query engine with web integration, advanced memory, and improved embeddings"""

    def __init__(
            self,
            chroma_db_path: str = None,
            milvus_lite_path: str = None,
            embedding_model: str = None,
            milvus_collection_name: str = "langchain_collection"
    ):
        # Load configuration
        self.chroma_db_path = chroma_db_path or os.getenv("CHROMA_DB_PATH", "chroma_store")
        self.milvus_lite_path = milvus_lite_path or os.path.join(
            os.getenv("MILVUS_LITE_FOLDER", "milvus_store"),
            os.getenv("MILVUS_LITE_DB_FILE", "milvus_lite.db")
        )

        # Enhanced embedding model selection
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL",
                                                            PROGRAMMING_EMBEDDING_MODELS['programming-optimized'])
        self.current_embedding_type = 'programming-optimized'

        self.milvus_collection_name = milvus_collection_name

        # Initialize components
        self.embeddings = None
        self.chroma_db = None
        self.milvus_db = None
        self.llm = None

        # NEW: Enhanced components
        self.enhanced_memory = self._initialize_enhanced_mem0()
        self.web_processor = JinaWebProcessor()
        self.user_session_id = f"user_{int(time.time())}"  # Dynamic session

        # Programming language patterns (enhanced)
        self.language_patterns = {
            'python': [r'def\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'\.py\b', r'python',
                       r'django', r'flask', r'pandas', r'numpy', r'__init__', r'self\.'],
            'javascript': [r'function\s+\w+', r'const\s+\w+', r'let\s+\w+', r'var\s+\w+', r'\.js\b',
                           r'node\.js', r'react', r'angular', r'vue', r'=>', r'async\s+', r'await\s+'],
            'java': [
                r'public\s+class\s+\w+', r'private\s+\w+\s+\w+', r'WeakHashMap',
                r'HashMap', r'System\.out\.println', r'public\s+static\s+void\s+main',
                r'\.java\b', r'spring', r'maven', r'gradle', r'import\s+java\.',
                r'new\s+\w+\s*\(', r'\.get\s*\(', r'\.put\s*\(', r'@Override'
            ],
            'cpp': [r'#include\s*<.*>', r'int\s+main\s*\(', r'std::', r'\.cpp\b', r'\.hpp\b',
                    r'namespace\s+', r'using\s+namespace'],
            'c': [r'#include\s*<.*>', r'int\s+main\s*\(', r'printf\s*\(', r'\.c\b', r'\.h\b',
                  r'malloc\s*\(', r'free\s*\('],
            'sql': [r'SELECT\s+', r'FROM\s+', r'WHERE\s+', r'INSERT\s+INTO', r'CREATE\s+TABLE',
                    r'UPDATE\s+', r'DELETE\s+FROM', r'JOIN\s+'],
            'html': [r'<html', r'<div', r'<p>', r'<span', r'<!DOCTYPE', r'<head>', r'<body>'],
            'css': [r'\.[\w-]+\s*{', r'#[\w-]+\s*{', r':\s*\w+;', r'@media', r'display\s*:']
        }

        # Enhanced programming concepts
        self.programming_concepts = [
            'algorithm', 'data structure', 'array', 'list', 'dictionary', 'hash', 'tree', 'graph',
            'sorting', 'searching', 'recursion', 'iteration', 'loop', 'function', 'method', 'class',
            'object', 'inheritance', 'polymorphism', 'encapsulation', 'abstraction', 'interface',
            'database', 'sql', 'nosql', 'api', 'rest', 'microservices', 'docker', 'kubernetes',
            'testing', 'unit test', 'integration test', 'debugging', 'optimization', 'performance',
            'security', 'authentication', 'authorization', 'encryption', 'framework', 'library'
        ]

        # Cache for query results
        self.query_cache = {}

        logger.info("Enhanced ProgrammingQueryEngine initialized")

    def _initialize_enhanced_embeddings(self, model_type: str = None):
        """Enhanced embedding initialization with model selection"""
        if model_type:
            self.current_embedding_type = model_type
            if model_type in PROGRAMMING_EMBEDDING_MODELS:
                model_name = PROGRAMMING_EMBEDDING_MODELS[model_type]
            else:
                model_name = model_type  # Direct model name
        else:
            model_name = self.embedding_model

        if self.embeddings is None or model_name != getattr(self.embeddings, 'model_name', ''):
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},  # Change to 'cuda' if available
                    encode_kwargs={'normalize_embeddings': True}  # Better for similarity
                )
                self.embedding_model = model_name
                logger.info(f"Initialized enhanced embeddings with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {e}")
                # Fallback to default
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=PROGRAMMING_EMBEDDING_MODELS['current-default']
                )

    def _initialize_chroma(self):
        """Lazy initialization of Chroma DB"""
        if self.chroma_db is None:
            self._initialize_enhanced_embeddings()
            try:
                self.chroma_db = Chroma(
                    persist_directory=self.chroma_db_path,
                    embedding_function=self.embeddings,
                    collection_name="langchain"
                )
                logger.info("Initialized Chroma DB")
            except Exception as e:
                logger.warning(f"Failed to initialize Chroma: {e}")

    def _initialize_milvus(self):
        """Lazy initialization of Milvus"""
        if self.milvus_db is None:
            self._initialize_enhanced_embeddings()
            try:
                self.milvus_db = Milvus(
                    collection_name=self.milvus_collection_name,
                    connection_args={"uri": self.milvus_lite_path},
                    embedding_function=self.embeddings
                )
                logger.info("Initialized Milvus DB")
            except Exception as e:
                logger.warning(f"Failed to initialize Milvus: {e}")

    def _initialize_groq_llm(self):
        """Initialize Groq LLM"""
        if self.llm is None:
            try:
                groq_api_key = os.getenv("GROQ_API_KEY")
                if groq_api_key:
                    self.llm = ChatGroq(
                        temperature=0.1,
                        model_name="llama3-8b-8192",
                        groq_api_key=groq_api_key,
                        max_tokens=2048  # Increased for better responses
                    )
                    logger.info("Initialized Groq LLM")
                else:
                    logger.warning("No Groq API key found")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM: {e}")

    def _initialize_enhanced_mem0(self):
        """Initialize enhanced memory system"""
        try:
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "enhanced_programming_memory",
                        "path": self.chroma_db_path,
                        # "metadata": {"hnsw:space": "cosine"}
                    },
                }
            }
            return EnhancedProgrammingMemory(config)
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced memory: {e}")
            return None

    # NEW: Embedding model comparison and switching
    def compare_embedding_models(self, query: str, models: List[str] = None) -> Dict[str, Any]:
        """Compare different embedding models on the same query"""
        if not models:
            models = ['fast-general', 'code-focused', 'programming-optimized']

        results = {}
        original_model = self.current_embedding_type

        for model_type in models:
            try:
                start_time = time.time()
                self._initialize_enhanced_embeddings(model_type)
                self._initialize_chroma()  # Reinitialize with new embeddings

                # Quick search to test
                if self.chroma_db:
                    test_results = self.chroma_db.similarity_search(query, k=5)
                    end_time = time.time()

                    results[model_type] = {
                        'query_time': end_time - start_time,
                        'num_results': len(test_results),
                        'model_name': PROGRAMMING_EMBEDDING_MODELS.get(model_type, model_type)
                    }
            except Exception as e:
                results[model_type] = {'error': str(e)}

        # Restore original model
        self._initialize_enhanced_embeddings(original_model)
        self._initialize_chroma()
        return results

    def switch_embedding_model(self, model_type: str):
        """Switch to a different embedding model"""
        if model_type in PROGRAMMING_EMBEDDING_MODELS:
            self._initialize_enhanced_embeddings(model_type)
            # Clear cache since model changed
            self.query_cache.clear()
            logger.info(f"Switched to {model_type} embeddings")
        else:
            logger.error(f"Unknown model type: {model_type}")

    # NEW: Web-augmented search
    def _web_augmented_search(self, query: str, config: ProgrammingQueryConfig) -> List[Document]:
        """Augment local results with relevant web content"""
        if not config.enable_web_search:
            return []

        try:
            # Generate web search URLs based on query
            search_urls = self._generate_search_urls(query)

            # Fetch and process web content
            web_documents = self.web_processor.process_programming_urls(search_urls)

            # Filter for programming relevance and limit results
            relevant_docs = [doc for doc in web_documents
                             if doc.metadata.get('programming_related', False)]

            return relevant_docs[:config.max_web_results]
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def _generate_search_urls(self, query: str) -> List[str]:
        """Generate relevant URLs for programming queries"""
        # Basic search URLs - you can enhance these based on detected languages

        base_urls = [
            "stackoverflow.com",
            "github.com",
            "developer.mozilla.org",
            "geeksforgeeks.org",
            "docs.oracle.com",
            "w3schools.com",
            "kotlinlang.org",
            "spring.io",
            # f"https://stackoverflow.com/search?q={query.replace(' ', '+')}",
        ]
        site_filter = "+OR+".join(f"site:{s}" for s in base_urls)
        f"https://www.google.com/search?q={query.replace(' ', '+')}+{site_filter}"

        # base_urls = [
        #     f"https://stackoverflow.com/search?q={query.replace(' ', '+')}",
        # ]

        # Add language-specific documentation
        query_lower = query.lower()
        if any(lang in query_lower for lang in ['spring framework', 'springboot']):
            base_urls.append("https://docs.spring.io/spring-framework/reference/index.html")
        elif any(lang in query_lower for lang in ['javascript', 'js', 'node']):
            base_urls.append("https://developer.mozilla.org/en-US/docs/Web/JavaScript")
        elif 'java' in query_lower:
            base_urls.append("https://docs.oracle.cspring docs"
                             "om/en/java/")

        return base_urls[:2]  # Limit to avoid rate limits

    def _detect_programming_languages(self, query: str, documents: List[Document]) -> List[str]:
        """Enhanced language detection"""
        detected = set()

        # Check query
        query_lower = query.lower()
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected.add(lang)

        # Check documents
        for doc in documents:
            content_lower = doc.page_content.lower()
            for lang, patterns in self.language_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content_lower, re.IGNORECASE):
                        detected.add(lang)

        return list(detected)

    def _extract_code_snippets(self, documents: List[Document]) -> List[str]:
        """Enhanced code snippet extraction"""
        code_snippets = []

        for doc in documents:
            content = doc.page_content

            # Look for code blocks (markdown format)
            code_blocks = re.findall(r'``````', content, re.DOTALL)
            code_snippets.extend([block.strip() for block in code_blocks])

            # Look for inline code
            inline_codes = re.findall(r'`([^`\n]+)`', content)
            code_snippets.extend(inline_codes)

            # Language-specific patterns
            for lang, patterns in self.language_patterns.items():
                for pattern in patterns:
                    if pattern.startswith(r'[a-zA-Z]'):  # Skip regex patterns
                        continue
                    matches = re.findall(pattern, content, re.DOTALL | re.MULTILINE)
                    code_snippets.extend(matches[:3])  # Limit per pattern

        return code_snippets[:20]  # Increased limit

    def _identify_concepts(self, query: str, documents: List[Document]) -> List[str]:
        """Enhanced concept identification"""
        identified = set()

        combined_text = f"{query} {' '.join([doc.page_content for doc in documents])}"
        combined_lower = combined_text.lower()

        for concept in self.programming_concepts:
            if concept in combined_lower:
                identified.add(concept)

        return list(identified)[:15]  # Increased limit

    def _generate_programming_queries(self, original_query: str) -> List[str]:
        """Enhanced query generation"""
        base_queries = [original_query]

        # Programming context variations
        variations = [
            f"how to {original_query}",
            f"{original_query} example",
            f"{original_query} code",
            f"{original_query} tutorial",
            f"{original_query} implementation",
            f"{original_query} best practices",
            f"{original_query} syntax",
            f"what is {original_query}",
            f"{original_query} error handling",
            f"{original_query} performance"
        ]

        # Detect language and add specific variations
        detected_lang = None
        for lang in ProgrammingLanguage:
            if lang.value in original_query.lower():
                detected_lang = lang.value
                break

        if detected_lang:
            variations.extend([
                f"{detected_lang} {original_query}",
                f"{original_query} in {detected_lang}",
                f"{detected_lang} {original_query} library",
                f"{detected_lang} {original_query} framework"
            ])

        base_queries.extend(variations[:10])  # Increased limit
        return base_queries

    # Enhanced search methods (keeping your existing ones but with improvements)
    def _similarity_search_both(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Search both databases with similarity"""
        results = []

        if self.chroma_db:
            try:
                chroma_results = self.chroma_db.similarity_search_with_score(query, k=top_k)
                results.extend([(doc, score) for doc, score in chroma_results])
            except Exception as e:
                logger.warning(f"Chroma search failed: {e}")

        if self.milvus_db:
            try:
                milvus_results = self.milvus_db.similarity_search_with_score(query, k=top_k)
                results.extend([(doc, score) for doc, score in milvus_results])
            except Exception as e:
                logger.warning(f"Milvus search failed: {e}")

        return results

    def _mmr_search_both(self, query: str, top_k: int, diversity: float) -> List[Document]:
        """MMR search on both databases"""
        results = []

        if self.chroma_db:
            try:
                chroma_results = self.chroma_db.max_marginal_relevance_search(
                    query, k=top_k, lambda_mult=diversity
                )
                results.extend(chroma_results)
            except Exception as e:
                logger.warning(f"Chroma MMR search failed: {e}")

        if self.milvus_db:
            try:
                milvus_results = self.milvus_db.max_marginal_relevance_search(
                    query, k=top_k, lambda_mult=diversity
                )
                results.extend(milvus_results)
            except Exception as e:
                logger.warning(f"Milvus MMR search failed: {e}")

        return results

    def _multi_query_search_both(self, query: str, top_k: int) -> List[Document]:
        """Multi-query search using Groq LLM"""
        if not self.llm:
            return []

        results = []
        if self.chroma_db:
            try:
                retriever = self.chroma_db.as_retriever(search_kwargs={"k": top_k})
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=retriever, llm=self.llm
                )
                chroma_results = multi_query_retriever.get_relevant_documents(query)
                results.extend(chroma_results)
            except Exception as e:
                logger.warning(f"Multi-query search failed: {e}")

        return results

    def _contextual_compression_search(self, query: str, top_k: int) -> List[Document]:
        """Contextual compression using Groq LLM"""
        if not self.llm:
            return []

        results = []
        if self.chroma_db:
            try:
                retriever = self.chroma_db.as_retriever(search_kwargs={"k": top_k * 2})
                compressor = LLMChainExtractor.from_llm(self.llm)
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=retriever
                )
                results = compression_retriever.get_relevant_documents(query)
            except Exception as e:
                logger.warning(f"Contextual compression failed: {e}")

        return results

    def _comprehensive_search(self, query: str, config: ProgrammingQueryConfig) -> ProgrammingQueryResult:
        """Enhanced comprehensive search with all new features"""
        start_time = time.time()

        # Initialize databases with selected embedding model
        self._initialize_enhanced_embeddings(config.embedding_model_type)
        self._initialize_chroma()
        self._initialize_milvus()
        self._initialize_groq_llm()

        all_results = {}
        strategies_used = []
        web_sources_count = 0
        memory_context_used = False
        personalization_strength = 0.0

        # 1. Basic similarity search
        try:
            similarity_results = self._similarity_search_both(query, config.top_k_per_strategy)
            for doc, score in similarity_results:
                doc_hash = hash(doc.page_content)
                if doc_hash not in all_results or all_results[doc_hash]['score'] < score:
                    all_results[doc_hash] = {
                        'doc': doc,
                        'score': score,
                        'strategies': ['similarity']
                    }
                else:
                    all_results[doc_hash]['strategies'].append('similarity')
            strategies_used.append('similarity')
            logger.info(f"Similarity search: {len(similarity_results)} results")
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")

        # 2. MMR search for diversity
        try:
            mmr_results = self._mmr_search_both(query, config.top_k_per_strategy, config.mmr_diversity)
            for doc in mmr_results:
                doc_hash = hash(doc.page_content)
                if doc_hash in all_results:
                    all_results[doc_hash]['score'] += 0.3
                    all_results[doc_hash]['strategies'].append('mmr')
                else:
                    all_results[doc_hash] = {
                        'doc': doc,
                        'score': 0.3,
                        'strategies': ['mmr']
                    }
            strategies_used.append('mmr')
            logger.info(f"MMR search: {len(mmr_results)} results")
        except Exception as e:
            logger.error(f"MMR search failed: {e}")

        # 3. Multi-query search if LLM available
        if self.llm:
            try:
                multi_query_results = self._multi_query_search_both(query, config.top_k_per_strategy)
                for doc in multi_query_results:
                    doc_hash = hash(doc.page_content)
                    if doc_hash in all_results:
                        all_results[doc_hash]['score'] += 0.4
                        all_results[doc_hash]['strategies'].append('multi_query')
                    else:
                        all_results[doc_hash] = {
                            'doc': doc,
                            'score': 0.4,
                            'strategies': ['multi_query']
                        }
                strategies_used.append('multi_query')
                logger.info(f"Multi-query search: {len(multi_query_results)} results")
            except Exception as e:
                logger.error(f"Multi-query search failed: {e}")

        # 4. Contextual compression if LLM available
        if self.llm:
            try:
                compression_results = self._contextual_compression_search(query, config.top_k_per_strategy)
                for doc in compression_results:
                    doc_hash = hash(doc.page_content)
                    if doc_hash in all_results:
                        all_results[doc_hash]['score'] += 0.5
                        all_results[doc_hash]['strategies'].append('contextual_compression')
                    else:
                        all_results[doc_hash] = {
                            'doc': doc,
                            'score': 0.5,
                            'strategies': ['contextual_compression']
                        }
                strategies_used.append('contextual_compression')
                logger.info(f"Contextual compression: {len(compression_results)} results")
            except Exception as e:
                logger.error(f"Contextual compression failed: {e}")

        # 5. Programming-specific query variations
        try:
            programming_queries = self._generate_programming_queries(query)
            for prog_query in programming_queries[1:6]:
                prog_results = self._similarity_search_both(prog_query, config.top_k_per_strategy // 2)
                for doc, score in prog_results:
                    doc_hash = hash(doc.page_content)
                    adjusted_score = score * 0.8
                    if doc_hash in all_results:
                        all_results[doc_hash]['score'] += adjusted_score * 0.2
                        if 'programming_variations' not in all_results[doc_hash]['strategies']:
                            all_results[doc_hash]['strategies'].append('programming_variations')
                    else:
                        all_results[doc_hash] = {
                            'doc': doc,
                            'score': adjusted_score * 0.2,
                            'strategies': ['programming_variations']
                        }
            strategies_used.append('programming_variations')
        except Exception as e:
            logger.error(f"Programming variations search failed: {e}")

        # NEW: 6. Web-augmented search
        if config.enable_web_search:
            try:
                web_documents = self._web_augmented_search(query, config)
                web_sources_count = len(web_documents)
                for doc in web_documents:
                    doc_hash = hash(doc.page_content)
                    if doc_hash in all_results:
                        all_results[doc_hash]['score'] += 0.6  # High weight for web sources
                        all_results[doc_hash]['strategies'].append('web_search')
                    else:
                        all_results[doc_hash] = {
                            'doc': doc,
                            'score': 0.6,
                            'strategies': ['web_search']
                        }
                if web_documents:
                    strategies_used.append('web_search')
                logger.info(f"Web search: {len(web_documents)} results")
            except Exception as e:
                logger.error(f"Web search failed: {e}")

        # Sort by combined score and take top results
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:config.final_top_k]

        # Extract final documents and scores
        final_documents = [result['doc'] for result in sorted_results]
        final_scores = [result['score'] for result in sorted_results]

        # Analyze results
        detected_languages = self._detect_programming_languages(query, final_documents)
        code_snippets = self._extract_code_snippets(final_documents)
        concepts = self._identify_concepts(query, final_documents)

        # Calculate confidence score
        confidence_score = min(1.0, sum(final_scores[:5]) / 5) if final_scores else 0.0

        query_time = time.time() - start_time

        # NEW: Enhanced synthesis with memory and personalization
        synthesized_answer = ""
        if self.llm and final_documents:
            if config.enhanced_memory and self.enhanced_memory:
                synthesized_answer = self._synthesize_answer_with_enhanced_memory(query, final_documents)
                memory_context_used = True
                personalization_strength = self.enhanced_memory.get_personalization_strength(query,
                                                                                             self.user_session_id)
            else:
                synthesized_answer = self._synthesize_answer_with_groq(query, final_documents)

        return ProgrammingQueryResult(
            synthesized_answer=synthesized_answer,
            documents=final_documents,
            scores=final_scores,
            detected_languages=detected_languages,
            code_snippets=code_snippets,
            concepts=concepts,
            related_topics=concepts[:5],
            query_strategies_used=strategies_used,
            query_time=query_time,
            total_results=len(all_results),
            confidence_score=confidence_score,
            web_sources_count=web_sources_count,
            memory_context_used=memory_context_used,
            embedding_model_used=self.embedding_model,
            personalization_strength=personalization_strength
        )


    def _synthesize_answer_with_groq(self, query: str, documents: List[Document]) -> str:
        """Enhanced synthesis using Groq AI"""
        # Combine all relevant document content
        context = "\n\n---DOCUMENT---\n\n".join([doc.page_content for doc in documents[:5]])

        # Extract code snippets specifically
        code_snippets = self._extract_code_snippets(documents)

        # Format code snippets for the prompt
        formatted_snippets = ""
        if code_snippets:
            # formatted_snippets = "\n".join([f"``````" for snippet in code_snippets[:8]])
            formatted_snippets = "\n".join([f"``````" for snippet in code_snippets[:8]])

        synthesis_prompt = f"""Based ONLY on the provided context from programming documentation, answer the user's question.

        CRITICAL RULES FOR CODE HANDLING:
        - ALWAYS include relevant code examples from the context
        - Preserve code formatting using proper markdown code blocks
        - Explain what each code example demonstrates
        - If code shows class definitions, method calls, or examples, include them
        - Use ``````java, etc. for syntax highlighting

        STRICT RULES:
        - Use ONLY information present in the context below
        - Include ALL relevant code examples found in the context
        - If code demonstrates the concept, show it prominently
        - Keep explanations clear and focused
        - If context lacks sufficient info, say "Based on the available documentation..."

        CONTEXT:
        {context}

        EXTRACTED CODE EXAMPLES:
        {formatted_snippets}

        QUESTION: {query}

        ANSWER (including relevant code examples from context):
        """

        try:
            response = self.llm.invoke(synthesis_prompt)
            answer = response.content.strip()

            # Ensure code is preserved
            # if code_snippets and '``` answer += f"\n\n**Example Code:**\n```\n{code_snippets[0][:500]}\n```
            if code_snippets and '```' not in answer:
                answer += f"\n\n**Example Code:**\n```java\n{code_snippets[0][:500]}\n```"

                return answer
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return "Unable to synthesize answer from the retrieved context."

    def _synthesize_answer_with_enhanced_memory(self, query: str, documents: List[Document]) -> str:
        """Enhanced synthesis using advanced memory features"""
        if not self.enhanced_memory:
            return self._synthesize_answer_with_groq(query, documents)

        try:
            # Get personalized context
            personal_context = self.enhanced_memory.get_personalized_context(query, self.user_session_id)

            # Store current query and documents in memory
            mock_result = type('obj', (object,), {
                'detected_languages': self._detect_programming_languages(query, documents),
                'concepts': self._identify_concepts(query, documents),
                'code_snippets': self._extract_code_snippets(documents),
                'confidence_score': 0.85,
                'query_strategies_used': ['enhanced_memory']
            })

            self.enhanced_memory.store_programming_context(query, mock_result, self.user_session_id)

            # Build context-aware prompt
            context_info = ""
            if personal_context['context_strength'] > 3:
                context_info = f"""
Based on your previous interactions, I notice you often work with:
- Languages: {', '.join(personal_context.get('language_preferences', [])[:3])}
- Related concepts you've explored: {', '.join(personal_context.get('related_concepts', [])[:3])}
"""

            # Enhanced synthesis prompt
            document_context = "\n\n---DOCUMENT---\n\n".join([doc.page_content for doc in documents[:5]])

            synthesis_prompt = f"""
{context_info}

Based on the programming documentation below and your previous learning patterns, provide a comprehensive answer.

DOCUMENTATION CONTEXT:
{document_context}

QUESTION: {query}

PERSONALIZED ANSWER (considering your typical interests and programming patterns):
"""

            response = self.llm.invoke(synthesis_prompt)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Enhanced memory synthesis failed: {e}")
            return self._synthesize_answer_with_groq(query, documents)

    def query(self, query: str, config: ProgrammingQueryConfig = None) -> ProgrammingQueryResult:
        """Main enhanced query method"""
        if config is None:
            config = ProgrammingQueryConfig()

        # Check cache
        cache_key = f"{hash(query)}_{hash(str(config))}"
        if config.enable_caching and cache_key in self.query_cache:
            logger.info("Returning cached result")
            return self.query_cache[cache_key]

        try:
            result = self._comprehensive_search(query, config)

            # Cache result
            if config.enable_caching:
                self.query_cache[cache_key] = result

            logger.info(f"Enhanced query completed: {len(result.documents)} results in {result.query_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Enhanced engine statistics"""
        stats = {
            'chroma_initialized': self.chroma_db is not None,
            'milvus_initialized': self.milvus_db is not None,
            'groq_llm_available': self.llm is not None,
            'embeddings_model': self.embedding_model,
            'current_embedding_type': self.current_embedding_type,
            'cache_size': len(self.query_cache),
            'supported_languages': len(ProgrammingLanguage),
            'programming_concepts': len(self.programming_concepts),
            'enhanced_memory_available': self.enhanced_memory is not None,
            'web_processor_available': self.web_processor is not None,
            'available_embedding_models': list(PROGRAMMING_EMBEDDING_MODELS.keys())
        }

        # Get collection stats
        try:
            if self.chroma_db:
                chroma_collection = self.chroma_db._collection
                stats['chroma_count'] = chroma_collection.count()
        except:
            stats['chroma_count'] = 'unknown'

        try:
            if self.milvus_db:
                stats['milvus_count'] = 'unknown'  # Placeholder
        except:
            stats['milvus_count'] = 'unknown'

        return stats


# Convenience function
def create_programming_query_engine(**kwargs) -> ProgrammingQueryEngine:
    """Factory function to create enhanced programming query engine"""
    return ProgrammingQueryEngine(**kwargs)
