# # src/knowledge.py
# import os
# import json
# from pathlib import Path
# from typing import Optional, List

# from bs4 import BeautifulSoup
# import requests

# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.tools import DuckDuckGoSearchRun

# DATA_DIR = Path("data")
# BANK_DIR = DATA_DIR / "banking"
# INDEX_DIR = BANK_DIR / "faiss_index"
# SOURCES_JSON = BANK_DIR / "bank_sources.json"

# class BankingKB:
#     def __init__(self) -> None:
#         self._emb = None
#         self._index: Optional[FAISS] = None

#     # ---------- public API ----------
#     def query(self, question: str) -> str:
#         idx = self._ensure_index()
#         hits = idx.similarity_search(question, k=6)
#         parts: List[str] = []
#         for i, h in enumerate(hits, 1):
#             src = h.metadata.get("source", "source")
#             parts.append(f"[{i}] {src}\n{h.page_content}")
#         return "\n\n".join(parts) if parts else "No relevant passages found in the local corpus."

#     def web_search(self, query: str) -> str:
#         return DuckDuckGoSearchRun().run(tool_input=query)

#     # ---------- internals ----------
#     def _ensure_index(self) -> FAISS:
#         if self._index is not None:
#             return self._index
#         if INDEX_DIR.exists():
#             self._index = FAISS.load_local(
#                 str(INDEX_DIR),
#                 embeddings=self._get_embeddings(),
#                 allow_dangerous_deserialization=True,
#             )
#             return self._index
#         return self._build_index()

#     def _get_embeddings(self):
#         if self._emb is None:
#             self._emb = OpenAIEmbeddings()
#         return self._emb

#     def _build_index(self) -> FAISS:
#         BANK_DIR.mkdir(parents=True, exist_ok=True)
#         splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)

#         docs = []

#         # PDFs
#         for pdf in BANK_DIR.glob("*.pdf"):
#             try:
#                 pages = PyPDFLoader(str(pdf)).load()
#                 for p in pages:
#                     p.metadata = {"source": pdf.name}
#                 docs.extend(pages)
#             except Exception:
#                 pass

#         # Text/Markdown
#         for textf in list(BANK_DIR.glob("*.txt")) + list(BANK_DIR.glob("*.md")):
#             try:
#                 raw = textf.read_text(encoding="utf8")
#                 chunks = splitter.split_text(raw)
#                 from langchain.schema import Document
#                 for c in chunks:
#                     docs.append(Document(page_content=c, metadata={"source": textf.name}))
#             except Exception:
#                 pass

#         if not docs:
#             raise RuntimeError(
#                 "No knowledge files in data/banking/. "
#                 "Run `python src/ingest_banking.py` to fetch official SBI/RBI/NPCI sources."
#             )

#         pieces = splitter.split_documents(docs)
#         idx = FAISS.from_documents(pieces, self._get_embeddings())
#         INDEX_DIR.mkdir(parents=True, exist_ok=True)
#         idx.save_local(str(INDEX_DIR))
#         self._index = idx
#         return idx













# src/knowledge.py
import os
import json
import asyncio
import datetime
from pathlib import Path
from typing import Optional, List

from bs4 import BeautifulSoup  # kept for parity with your ingest approach
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import Document  # works across LC versions

DATA_DIR = Path("data")
BANK_DIR = DATA_DIR / "banking"
INDEX_DIR = BANK_DIR / "faiss_index"
SOURCES_JSON = BANK_DIR / "bank_sources.json"


class BankingKB:
    def __init__(self) -> None:
        self._emb: Optional[OpenAIEmbeddings] = None
        self._index: Optional[FAISS] = None

    # ---------- public API ----------
    def query(self, question: str) -> str:
        idx = self._ensure_index()
        hits = idx.similarity_search(question, k=6)
        parts: List[str] = []
        for i, h in enumerate(hits, 1):
            src = h.metadata.get("source", "source")
            parts.append(f"[{i}] {src}\n{h.page_content}")
        return "\n\n".join(parts) if parts else "No relevant passages found in the local corpus."

    def web_search(self, query: str) -> str:
        # Synchronous fallback (unused by agent after async addition, but kept for compatibility)
        return DuckDuckGoSearchRun().run(tool_input=query)

    async def web_search_async(self, query: str) -> str:
        loop = asyncio.get_event_loop()

        def _run():
            return DuckDuckGoSearchRun().run(tool_input=query)

        txt = await loop.run_in_executor(None, _run)
        stamp = datetime.datetime.now().strftime("%b %d, %Y")
        return f"(web search as of {stamp})\n{txt}"

    # ---------- internals ----------
    def _ensure_index(self) -> FAISS:
        if self._index is not None:
            return self._index
        if INDEX_DIR.exists():
            self._index = FAISS.load_local(
                str(INDEX_DIR),
                embeddings=self._get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            return self._index
        return self._build_index()

    def _get_embeddings(self):
        if self._emb is None:
            # Reads OPENAI_API_KEY from env; no extra config needed
            self._emb = OpenAIEmbeddings()
        return self._emb

    def _build_index(self) -> FAISS:
        BANK_DIR.mkdir(parents=True, exist_ok=True)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)

        docs: List[Document] = []

        # PDFs
        for pdf in BANK_DIR.glob("*.pdf"):
            try:
                pages = PyPDFLoader(str(pdf)).load()
                for p in pages:
                    p.metadata = {"source": pdf.name}
                docs.extend(pages)
            except Exception:
                # skip unreadable PDFs quietly
                pass

        # Text/Markdown
        for textf in list(BANK_DIR.glob("*.txt")) + list(BANK_DIR.glob("*.md")):
            try:
                raw = textf.read_text(encoding="utf8")
                chunks = splitter.split_text(raw)
                for c in chunks:
                    docs.append(Document(page_content=c, metadata={"source": textf.name}))
            except Exception:
                pass

        if not docs:
            raise RuntimeError(
                "No knowledge files in data/banking/. "
                "Run `python src/ingest_banking.py` to fetch official SBI/RBI/NPCI sources."
            )

        pieces = splitter.split_documents(docs)
        idx = FAISS.from_documents(pieces, self._get_embeddings())
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        idx.save_local(str(INDEX_DIR))
        self._index = idx
        return idx
#test
