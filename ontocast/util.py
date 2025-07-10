import hashlib
import logging

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


def truncate_ontology_string(ontology_str: str, max_chars: int = 50000) -> str:
    """Truncate ontology string to prevent API limits being exceeded.

    Args:
        ontology_str: The full ontology string in turtle format.
        max_chars: Maximum number of characters to keep (default: 50000).

    Returns:
        str: Truncated ontology string with truncation notice if needed.
    """
    if len(ontology_str) <= max_chars:
        return ontology_str

    # Find a good truncation point (end of a statement)
    truncated = ontology_str[:max_chars]

    # Try to find the last complete statement ending with '.'
    last_dot = truncated.rfind(".")
    if last_dot > max_chars * 0.8:  # Only use if it's not too far back
        truncated = truncated[: last_dot + 1]

    logger.warning(
        f"Ontology string truncated from {len(ontology_str)} to {len(truncated)} characters"
    )

    # Add truncation notice
    truncated += f"\n\n# ... [TRUNCATED: {len(ontology_str) - len(truncated)} characters omitted] ..."

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
