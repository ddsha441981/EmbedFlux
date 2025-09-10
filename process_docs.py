import os
import json
import hashlib
import logging
from typing import List, Dict, Tuple
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Milvus imports
from langchain_milvus import Milvus
from mem0 import Memory

# ---------- Logging Configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)



# Enhanced embedding models (same as in query engine)
PROGRAMMING_EMBEDDING_MODELS = {
    'fast-general': 'sentence-transformers/all-MiniLM-L12-v2',
    'code-focused': 'microsoft/codebert-base',
    'programming-optimized': 'sentence-transformers/all-MiniLM-L12-v2',
    'current-default': 'sentence-transformers/all-mpnet-base-v2'
}

# Update the embedding model selection
DEFAULT_EMBEDDING_TYPE = os.getenv("DEFAULT_EMBEDDING_TYPE", "programming-optimized")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", PROGRAMMING_EMBEDDING_MODELS[DEFAULT_EMBEDDING_TYPE])

# ---------- Configuration Management ----------
load_dotenv()  # Load .env file if present

DEFAULT_DOCS_PATH = os.getenv("DOCS_PATH", "docs")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_store")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

PROCESSED_HASH_FOLDER = os.getenv("PROCESSED_HASH_FOLDER", "metadata")
PROCESSED_FILES_HASH_JSON = os.getenv("PROCESSED_FILES_HASH_JSON", "processed_files_hash.json")
PROCESSED_FILES_HASH_PATH = os.path.join(PROCESSED_HASH_FOLDER, PROCESSED_FILES_HASH_JSON)

# Milvus Lite config - use folder path instead of single file
MILVUS_LITE_FOLDER = os.getenv("MILVUS_LITE_FOLDER", "milvus_store")
MILVUS_LITE_DB_FILE = os.getenv("MILVUS_LITE_DB_FILE", "milvus_lite.db")
MILVUS_LITE_PATH = os.path.join(MILVUS_LITE_FOLDER, MILVUS_LITE_DB_FILE)
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "langchain_collection")

# Ensure required folders exist
os.makedirs(PROCESSED_HASH_FOLDER, exist_ok=True)
os.makedirs(MILVUS_LITE_FOLDER, exist_ok=True)







# ---------- Mem0 Configuration ----------
def get_mem0_config():
    """Get Mem0 configuration using your existing vector stores"""
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": os.getenv("MEM0_COLLECTION_NAME", "programming_memory"),
                "path": CHROMA_DB_PATH,  # Use the global variable instead of self.chroma_db_path
            },
        }
        # No LLM configuration needed - this prevents OpenAI API key errors
    }


def populate_memory_from_documents(documents: List, memory_instance, user_id="system"):
    """Enhanced Mem0 population with better concept extraction"""
    if not memory_instance:
        return

    programming_indicators = [
        'class', 'function', 'method', 'algorithm', 'data structure',
        'pattern', 'best practice', 'optimization', 'error handling'
    ]

    for doc in documents:
        content = doc.page_content
        content_lower = content.lower()

        # More sophisticated concept detection
        found_concepts = [indicator for indicator in programming_indicators
                          if indicator in content_lower]

        if found_concepts:
            # Store with richer metadata
            concept_summary = f"Programming concepts: {', '.join(found_concepts[:3])}. Content: {content[:800]}"
            memory_instance.add(
                concept_summary,
                user_id=user_id,
                metadata={'concepts': found_concepts, 'source': 'document_processing'}
            )


# def populate_memory_from_documents(documents: List, memory_instance, user_id="system"):
#     """Populate Mem0 with processed document knowledge"""
#
#     if not memory_instance:
#         return
#
#     for doc in documents:
#         # Extract key programming concepts and store in memory
#         content = doc.page_content
#
#         # Create memory entries for important concepts
#         if any(keyword in content.lower() for keyword in ['class', 'function', 'method', 'algorithm']):
#             memory_instance.add(
#                 f"Programming concept: {content[:800]}",
#                 user_id=user_id
#             )


# ---------- Utility Functions ----------
def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of the file content."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error computing hash for {file_path}: {e}")
        return ""

# ---------- Metadata Management ----------
def load_processed_files_hash() -> Dict[str, str]:
    """Load dict of filename to file hash."""
    if os.path.exists(PROCESSED_FILES_HASH_PATH):
        try:
            with open(PROCESSED_FILES_HASH_PATH, "r") as f:
                data = json.load(f)
                logging.info(f"Loaded processed file hashes for {len(data)} files from {PROCESSED_FILES_HASH_PATH}.")
                return data
        except Exception as e:
            logging.warning(f"Could not read processed files hash metadata: {e}")
    else:
        logging.info(f"No processed files hash metadata file found at {PROCESSED_FILES_HASH_PATH}.")
    return {}

def save_processed_files_hash(processed_hashes: Dict[str, str]):
    """Save dict of filename to file hash."""
    try:
        with open(PROCESSED_FILES_HASH_PATH, "w") as f:
            json.dump(processed_hashes, f, indent=2)
        logging.info(f"Saved processed files hash metadata for {len(processed_hashes)} files at {PROCESSED_FILES_HASH_PATH}.")
    except Exception as e:
        logging.error(f"Could not save processed files hash metadata: {e}")

# ---------- Core Logic with hash-based check ----------
def load_documents_with_hash_check(directory_path: str, processed_hashes: Dict[str, str]) -> Tuple[List, Dict[str, str]]:
    """Load documents which are new or changed based on SHA256 hash."""
    documents = []
    updated_hashes = {}

    try:
        files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    except Exception as e:
        logging.error(f"Failed to list files in {directory_path}: {e}")
        raise

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        current_hash = compute_file_hash(file_path)
        if current_hash == "":
            logging.warning(f"Skipping file with hash error: {filename}")
            continue

        prev_hash = processed_hashes.get(filename)
        if prev_hash == current_hash:
            logging.info(f"Skipping unchanged file: {filename}")
            continue

        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
            updated_hashes[filename] = current_hash
            logging.info(f"Loaded new/modified document: {filename}")
        except Exception as e:
            logging.error(f"Failed to load document {filename}: {e}")

    return documents, updated_hashes

def split_documents(documents: List, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logging.info(f"Documents split into {len(chunks)} chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
    return chunks

def embed_and_store_both(
        docs: List,
        chroma_db_path: str = CHROMA_DB_PATH,
        milvus_lite_path: str = MILVUS_LITE_PATH,
        milvus_collection_name: str = MILVUS_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
        store_in_chroma: bool = True,
        store_in_milvus: bool = True
):
    """Embed document chunks and store them in both Chroma and Milvus Lite vector DBs."""
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if store_in_chroma:
        chroma_db = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=chroma_db_path
        )
        logging.info(f"Embeddings stored in Chroma vector DB at: {chroma_db_path}")

    if store_in_milvus:
        # For Milvus Lite: use connection_args with uri pointing to folder-based file
        milvus_db = Milvus.from_documents(
            docs,
            embeddings,
            collection_name=milvus_collection_name,
            connection_args={"uri": milvus_lite_path}
        )
        logging.info(f"Embeddings stored in Milvus Lite at: {milvus_lite_path}")

    return None

# def process_and_store(
#         directory_path: str = DEFAULT_DOCS_PATH,
#         chroma_db_path: str = CHROMA_DB_PATH,
#         milvus_lite_path: str = MILVUS_LITE_PATH,
#         milvus_collection_name: str = MILVUS_COLLECTION_NAME,
#         model_name: str = EMBEDDING_MODEL,
#         chunk_size: int = CHUNK_SIZE,
#         chunk_overlap: int = CHUNK_OVERLAP
# ):
#     """End-to-end pipeline for processing, embedding, and storing documents,
#        skipping unchanged files by comparing hashes, storing embeddings in Chroma and Milvus Lite."""
#     try:
#         processed_hashes = load_processed_files_hash()
#
#         documents, updated_hashes = load_documents_with_hash_check(directory_path, processed_hashes)
#
#         if not documents:
#             logging.info("No new or modified documents to process. Exiting.")
#             return
#
#         chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         if not chunks:
#             logging.warning("No chunks created. Exiting.")
#             return
#
#         if chunks:
#             try:
#                 # Initialize Mem0 (same config as query engine)
#                 try:
#                     mem0_config = get_mem0_config()
#                     memory = Memory.from_config(mem0_config)  # Define this config
#                     populate_memory_from_documents(chunks, memory)
#                     logging.info("Documents added to Mem0 memory")
#                 except Exception as e:
#                     logging.warning(f"Failed to populate Mem0: {e}")
#
#         embed_and_store_both(
#             chunks,
#             chroma_db_path=chroma_db_path,
#             milvus_lite_path=milvus_lite_path,
#             milvus_collection_name=milvus_collection_name,
#             model_name=model_name
#         )
#
#         processed_hashes.update(updated_hashes)
#         save_processed_files_hash(processed_hashes)
#
#         logging.info("Workflow completed successfully.")
#     except Exception as e:
#         logging.error(f"Pipeline error: {e}")
#         raise

def process_and_store(
        directory_path: str = DEFAULT_DOCS_PATH,
        chroma_db_path: str = CHROMA_DB_PATH,
        milvus_lite_path: str = MILVUS_LITE_PATH,
        milvus_collection_name: str = MILVUS_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
):
    """End-to-end pipeline for processing, embedding, and storing documents,
       skipping unchanged files by comparing hashes, storing embeddings in Chroma and Milvus Lite."""
    try:
        processed_hashes = load_processed_files_hash()

        documents, updated_hashes = load_documents_with_hash_check(directory_path, processed_hashes)

        if not documents:
            logging.info("No new or modified documents to process. Exiting.")
            return

        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            logging.warning("No chunks created. Exiting.")
            return

        # Store in vector databases first
        embed_and_store_both(
            chunks,
            chroma_db_path=chroma_db_path,
            milvus_lite_path=milvus_lite_path,
            milvus_collection_name=milvus_collection_name,
            model_name=model_name
        )

        # Then populate Mem0 memory
        if chunks:
            try:
                # Initialize Mem0 with proper config
                mem0_config = get_mem0_config()  # Use the function we defined
                memory = Memory.from_config(mem0_config)
                populate_memory_from_documents(chunks, memory)
                logging.info("Documents added to Mem0 memory")
            except Exception as e:
                logging.warning(f"Failed to populate Mem0: {e}")

        processed_hashes.update(updated_hashes)
        save_processed_files_hash(processed_hashes)

        logging.info("Workflow completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        raise



def populate_memory_from_documents(documents: List, memory_instance, user_id="system"):
    """Populate Mem0 with processed document knowledge"""

    if not memory_instance:
        return

    for doc in documents:
        # Extract key programming concepts and store in memory
        content = doc.page_content

        # Create memory entries for important concepts
        if any(keyword in content.lower() for keyword in ['class', 'function', 'method', 'algorithm']):
            memory_instance.add(
                f"Programming concept: {content[:800]}",
                user_id=user_id
            )
if __name__ == "__main__":
    process_and_store()

