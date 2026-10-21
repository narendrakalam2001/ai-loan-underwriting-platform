# ============================================================
# RAG ENGINE — Module 5A (Policy Intelligence Engine)
# ============================================================
# Deliberately does NOT use ChromaDB or LangChain.
#
#   - ChromaDB 0.5.5 and LangChain 0.2.x both hard-pin `numpy<2.0`
#     (confirmed via `pip check` in this project's own environment).
#     This platform's fraud-detection model requires `numpy>=2.0`
#     (see DAY2/3 notes — a numpy<2 environment breaks unpickling the
#     MLP champion's RandomState). Pulling in either library would
#     silently reintroduce that exact bug.
#   - Both also pull in large ML stacks (chromadb bundles onnxruntime;
#     langchain pulls numpy-pinned sub-packages) just to do retrieval.
#
# Instead: a small TF-IDF + cosine-similarity retriever built entirely
# on scikit-learn, which this project already depends on. No GPU, no
# model download, no new dependency, no numpy-version risk. For a
# knowledge base this size (a few hundred KB of policy text), TF-IDF
# retrieval quality is adequate — see README Module 5 notes for when
# to upgrade to real embeddings instead.
# ============================================================

import os
import re
import json
import glob
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_KB_DIR    = os.path.join("data", "knowledge_base")
DEFAULT_INDEX_PATH = os.path.join("data", "knowledge_base", "_index.joblib")


@dataclass
class RetrievedChunk:
    text:   str
    source: str
    score:  float

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text: str, source: str, chunk_size: int = 380, overlap: int = 80) -> List[Dict]:
    """
    Structure-aware chunker.

      1. Blocks = text split on blank lines.
      2. A short block with no sentence-ending punctuation is a HEADING and
         is attached to the block that follows it ("4. DELINQUENCY HISTORY
         POLICY" + its rules), instead of becoming a useless chunk of its own.
      3. A "Q12: ... A12: ..." FAQ pair is ATOMIC: never packed with a
         neighbour, so retrieval returns one question with its own answer.
      4. Other small blocks are packed together up to `chunk_size`.
      5. A block longer than `chunk_size` is split at sentence/rule
         boundaries (not mid-word); every piece keeps its heading as a
         prefix. Only a single sentence longer than `chunk_size` is
         hard-split with an `overlap`-character window.

    An earlier pure sliding-window version cut policy sections mid-rule and
    blended unrelated FAQs, so e.g. "3 serious delinquencies" retrieved the
    bare heading instead of the rule that answers it.
    """
    raw_blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", text)]
    raw_blocks = [b for b in raw_blocks if b]
    if not raw_blocks:
        return []

    def is_heading(b: str) -> bool:
        return len(b) <= 90 and not re.search(r"[.!?:]\s*$", b) and not re.match(r"Q\d+:", b)

    # (heading, body) units
    units, pending_heading = [], ""
    for b in raw_blocks:
        if is_heading(b):
            pending_heading = (pending_heading + " " + b).strip() if pending_heading else b
            continue
        units.append((pending_heading, b))
        pending_heading = ""
    if pending_heading:
        units.append(("", pending_heading))

    chunks: List[Dict] = []
    current, current_heading = "", ""

    def emit(txt: str):
        txt = txt.strip()
        if txt:
            chunks.append({"text": txt, "source": source})

    def flush():
        nonlocal current, current_heading
        emit(current)
        current, current_heading = "", ""

    def hard_split(txt: str):
        start = 0
        while start < len(txt):
            end_ = min(start + chunk_size, len(txt))
            emit(txt[start:end_])
            if end_ >= len(txt):
                break
            start = end_ - overlap

    for heading, body in units:
        prefix = (heading + " — ") if heading else ""

        if re.match(r"Q\d+:", body):                       # atomic FAQ pair
            flush()
            if len(prefix + body) <= chunk_size * 2:
                emit(prefix + body)
            else:
                hard_split(prefix + body)
            continue

        if len(prefix + body) > chunk_size:                # long section → sentence split
            flush()
            sentences = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", body) if x.strip()]
            piece = ""
            for sent in sentences:
                if len(prefix + sent) > chunk_size:
                    if piece:
                        emit(prefix + piece)
                        piece = ""
                    hard_split(prefix + sent)
                elif len(prefix + piece) + len(sent) + 1 > chunk_size and piece:
                    emit(prefix + piece)
                    piece = sent
                else:
                    piece = (piece + " " + sent).strip()
            if piece:
                emit(prefix + piece)
            continue

        if current and len(current) + len(prefix + body) + 1 > chunk_size:
            flush()
        current = (current + " " + prefix + body).strip()
    flush()
    return chunks


# ── Light stemming analyzer ────────────────────────────────────────
# Plain TF-IDF is purely lexical, so "delinquencies" never matches
# "delinquency". A crude suffix-stripper is enough to fix the common
# plural/verb-form misses without pulling in NLTK/spaCy. Must be a
# module-level function so the fitted vectorizer stays picklable.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(ENGLISH_STOP_WORDS)


def _stem(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ies"):
        return tok[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def stem_analyzer(text: str) -> List[str]:
    # Pure-digit tokens are dropped: section numbers ("7. PRICING", "4.1") would otherwise
    # match a digit in the user's question ("I have 7 dependents") and outrank the real answer.
    toks = [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and not t.isdigit()]
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


# ============================================================
# KNOWLEDGE BASE — fit / save / load / query
# ============================================================

class KnowledgeBase:
    """
    Fit a TF-IDF index over a set of text chunks, persist it, and query
    it by cosine similarity. Deliberately simple: no external vector DB,
    no server process, just a joblib file.
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.chunks: List[Dict] = []

    def fit(self, chunks: List[Dict]):
        if not chunks:
            raise ValueError("Cannot fit a KnowledgeBase on zero chunks")

        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        self.vectorizer = TfidfVectorizer(
            analyzer=stem_analyzer,
            max_features=20000,
            min_df=1,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

        logger.info("KnowledgeBase fit | chunks=%d | vocab=%d",
                    len(chunks), len(self.vectorizer.vocabulary_))
        return self

    def query(self, question: str, top_k: int = 4, min_score: float = 0.05) -> List[RetrievedChunk]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("KnowledgeBase not fitted/loaded — call fit() or load() first")

        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in top_idx:
            if sims[i] < min_score:
                continue
            results.append(RetrievedChunk(
                text=self.chunks[i]["text"],
                source=self.chunks[i]["source"],
                score=round(float(sims[i]), 4),
            ))

        logger.info("RAG query | q=%r | top_score=%.4f | hits=%d",
                    question[:60], float(sims[top_idx[0]]) if len(top_idx) else 0.0, len(results))
        return results

    def save(self, path: str = DEFAULT_INDEX_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
            "chunks": self.chunks,
        }, path)
        logger.info("KnowledgeBase saved → %s (%d chunks)", path, len(self.chunks))

    @classmethod
    def load(cls, path: str = DEFAULT_INDEX_PATH) -> "KnowledgeBase":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No knowledge base index at {path}. Run "
                f"`python scripts/ingest_knowledge_base.py` first."
            )
        payload = joblib.load(path)
        kb = cls()
        kb.vectorizer = payload["vectorizer"]
        kb.matrix = payload["matrix"]
        kb.chunks = payload["chunks"]
        logger.info("KnowledgeBase loaded ← %s (%d chunks)", path, len(kb.chunks))
        return kb


# ============================================================
# INGESTION — build the index from data/knowledge_base/*.txt|*.pdf
# ============================================================

def get_knowledge_base(path: str = DEFAULT_INDEX_PATH):
    """
    Loads the saved index, or returns None (not an exception) if it hasn't
    been built yet — callers (chatbot, decision explanations) should
    degrade gracefully rather than crash when Module 5 hasn't been set up
    (run `python scripts/ingest_knowledge_base.py` first).
    """
    try:
        return KnowledgeBase.load(path)
    except FileNotFoundError:
        logger.info("No knowledge base found at %s — faq_retriever/policy lookups unavailable", path)
        return None


def ingest_directory(kb_dir: str = DEFAULT_KB_DIR) -> KnowledgeBase:
    """
    Reads every .txt and .pdf file in `kb_dir`, chunks it, and fits a
    fresh KnowledgeBase. .pdf files are read with PyMuPDF (already a
    project dependency via document_parser.py) if present.
    """
    all_chunks: List[Dict] = []

    txt_paths = sorted(glob.glob(os.path.join(kb_dir, "*.txt")))
    pdf_paths = sorted(glob.glob(os.path.join(kb_dir, "*.pdf")))

    for path in txt_paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        logger.info("Ingested %s → %d chunks", source, len(chunks))

    for path in pdf_paths:
        try:
            import fitz
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            logger.warning("Could not read PDF %s: %s", path, e)
            continue
        source = os.path.basename(path)
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        logger.info("Ingested %s → %d chunks", source, len(chunks))

    if not all_chunks:
        raise RuntimeError(
            f"No .txt or .pdf files found in {kb_dir}/ — nothing to ingest. "
            f"See data/knowledge_base/README.md for what belongs here."
        )

    kb = KnowledgeBase().fit(all_chunks)
    return kb


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb = ingest_directory()
    kb.save()
    print(f"Knowledge base built: {len(kb.chunks)} chunks from "
          f"{len(set(c['source'] for c in kb.chunks))} document(s)")
