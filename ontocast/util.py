import hashlib
import logging
from typing import List, Optional

import torch
from langchain_community.utils.math import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from rdflib import Graph
from rdflib.namespace import NamespaceManager

logger = logging.getLogger(__name__)


def iri2namespace(iri: str, ontology: bool = False) -> str:
    """Convert an IRI to a namespace string.

    Args:
        iri: The IRI to convert.
        ontology: If True, append '#' for ontology namespace, otherwise '/'.

    Returns:
        str: The converted namespace string.
    """
    iri = iri.rstrip("#")
    return f"{iri}#" if ontology else f"{iri}/"


def get_rdflib_namespace_mappings() -> dict:
    g = Graph()
    ns_manager = NamespaceManager(g)
    return {str(uri): prefix for prefix, uri in ns_manager.namespaces()}


# Instead of truncating use semnatic chunking in the ChunkerTool class
def chunk_ontology_semantically(
    ontology_str: str,
    max_chars: int = 50000,
    embeddings=None,
    context: Optional[str] = None,
) -> list[str]:
    """Split an ontology string into semantic chunks using ChunkerTool.

    The import for ChunkerTool is done locally to avoid circular imports at
    module-import time.
    """

    from ontocast.tool.chunk.chunker import ChunkerTool  # local import

    chunker = ChunkerTool(
        breakpoint_threshold_amount=95,
        breakpoint_threshold_type="percentile",
        max_chunk_size=20000,
        # model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    chunks = chunker(ontology_str)
    return chunks


def _safe_embeddings(
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.warning("CUDA OOM; retrying on CPU")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
            )
        raise


# Instead of outputting all chunks, return only the semantically most relevant ones
def select_relevant_ontology_chunks(
    ontology_str: str,
    context: str,
    max_chunks: int = 3,
    max_chars: int = 50000,
    embeddings=None,
) -> list[str]:
    """Return the ontology chunks that are most relevant to a given context.

    This helper first splits the ontology into semantic chunks (via ``ChunkerTool``)
    and then ranks those chunks by cosine similarity to the *context* string
    using sentence-transformer embeddings.  Only the top ``max_chunks`` chunks are
    returned, and the function respects an overall character budget of
    ``max_chars``.

    Args:
        ontology_str: Complete ontology (Turtle) as a raw string.
        context: Natural-language text we want the ontology chunks to be relevant to.
        max_chunks: Maximum number of chunks to return (default: 3).
        max_chars: Combined character budget for the returned chunks (default: 50 000).
        embeddings: Optional pre-initialised ``Embeddings`` instance to reuse.

    Returns:
        A list of the most relevant chunk strings.
    """

    if embeddings is None:
        embeddings = _safe_embeddings()

    # 1. Chunk the ontology semantically
    chunks = chunk_ontology_semantically(
        ontology_str, max_chars=max_chars, embeddings=embeddings, context=context
    )

    if not chunks:
        return []

    # Shortcut: if we already meet the constraints, skip ranking
    if len(chunks) <= max_chunks and len(ontology_str) <= max_chars:
        return chunks

    # 2. Prepare an embedding model if none provided
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # 3. Embed context and chunks
    context_vector = embeddings.embed_query(context)
    chunk_vectors = embeddings.embed_documents(chunks)

    # 4. Rank chunks by similarity to the context
    similarities = [
        cosine_similarity([vec], [context_vector])[0][0] for vec in chunk_vectors
    ]
    ranked_chunks = [
        chunk
        for _, chunk in sorted(
            zip(similarities, chunks), key=lambda t: t[0], reverse=True
        )
    ]

    # 5. Select top-k within char budget
    selected: List[str] = []
    total_chars = 0
    for chunk in ranked_chunks:
        if len(selected) >= max_chunks:
            break
        if total_chars + len(chunk) > max_chars:
            break
        selected.append(chunk)
        total_chars += len(chunk)

    return selected


# -----------------------------------------------------------------------------
# Public helper used throughout the agents
# -----------------------------------------------------------------------------


def truncate(ontology_str: str, max_chars: int = 50000):
    truncated = ontology_str[:max_chars]

    # Try to end at the last full statement
    last_dot = truncated.rfind(".")
    if last_dot > max_chars * 0.8:
        truncated = truncated[: last_dot + 1]

    truncated += f"\n\n# ... [TRUNCATED: {len(ontology_str) - len(truncated)} characters omitted] ..."

    logger.warning(
        "Ontology string naive-truncated from %d to %d characters",
        len(ontology_str),
        len(truncated),
    )
    return truncated


def truncate_ontology_string(
    ontology_str: str,
    max_chars: int = 50000,
    embeddings=None,
    context: Optional[str] = None,
) -> str:
    """Return a shortened ontology string that fits within the character limit.

    If a ``context`` string is given the function will *semantically* trim the
    ontology using embeddings; otherwise it performs a simple character-based
    truncation.
    """

    if len(ontology_str) <= max_chars:
        return ontology_str

    # When we have context we prefer semantic selection
    if context is not None:
        selected_chunks = select_relevant_ontology_chunks(
            ontology_str,
            context=context,
            max_chars=max_chars,
            embeddings=embeddings,
        )
        ontology_str_sem = "\n\n".join(selected_chunks)
        logger.warning(
            "Ontology string was semantically reduced from %d to %d characters",
            len(ontology_str),
            len(ontology_str_sem),
        )
        if len(ontology_str_sem) > 0:
            return ontology_str_sem
        else:
            logger.warning(
                "Semantic selection returned no chunks; using naive truncation"
            )
            truncated = truncate(ontology_str, max_chars)
            truncated += f"\n\n# ... [TRUNCATED: {len(ontology_str) - len(truncated)} characters omitted] ..."
            logger.warning(
                "Ontology string naive-truncated from %d to %d characters",
                len(ontology_str),
                len(truncated),
            )
            return truncated

    # ------------------------------------------------------------------
    # Fallback: naïve truncation (keep behaviour identical to old version)
    # ------------------------------------------------------------------
    truncated = truncate(ontology_str, max_chars)
    truncated += f"\n\n# ... [TRUNCATED: {len(ontology_str) - len(truncated)} characters omitted] ..."

    logger.warning(
        "Ontology string naive-truncated from %d to %d characters",
        len(ontology_str),
        len(truncated),
    )
    return truncated


# TODO: Incorporate smart truncation
def truncate_text(text: str, max_chars: int = 30000) -> str:
    """Truncate text to prevent API limits being exceeded.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to keep (default: 30000).

    Returns:
        str: Truncated text with truncation notice if needed.
    """
    if len(text) <= max_chars:
        return text

    # Find a good truncation point (end of a sentence or paragraph)
    truncated = text[:max_chars]

    # Try to find the last complete sentence ending with '.', '!', or '?'
    last_sentence_end = max(
        truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?")
    )

    if last_sentence_end > max_chars * 0.8:  # Only use if it's not too far back
        truncated = truncated[: last_sentence_end + 1]

    logger.warning(f"Text truncated from {len(text)} to {len(truncated)} characters")

    # Add truncation notice
    truncated += f"\n\n[TRUNCATED: {len(text) - len(truncated)} characters omitted]"

    return truncated


def render_text_hash(text: str) -> str:
    """Render a hash of the text for use as a unique identifier.

    Args:
        text: The text to hash.

    Returns:
        str: The hash of the text.
    """
    return hashlib.md5(text.encode()).hexdigest()


CONVENTIONAL_MAPPINGS = {
    "http://www.w3.org/2002/07/owl": "owl",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns": "rdf",
    "http://www.w3.org/2000/01/rdf-schema": "rdfs",
    "http://www.w3.org/2001/XMLSchema": "xsd",
    "http://www.w3.org/2004/02/skos/core": "skos",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://schema.org/": "schema",
    "http://example.org/": "ex",
}
