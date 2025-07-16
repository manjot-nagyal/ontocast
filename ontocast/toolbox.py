import asyncio
import hashlib
import json
import pathlib
from typing import Optional

from langchain_core.prompts import PromptTemplate

from ontocast.dspy_output_parser import DspyOutputParser
from ontocast.onto import Ontology, OntologyProperties, RDFGraph
from ontocast.tool import (
    ChunkerTool,
    ConverterTool,
    FilesystemTripleStoreManager,
    FusekiTripleStoreManager,
    Neo4jTripleStoreManager,
    TripleStoreManager,
)
from ontocast.tool.aggregate import ChunkRDFGraphAggregator
from ontocast.tool.llm import LLMTool
from ontocast.tool.ontology_manager import OntologyManager


def update_ontology_properties(
    o: Ontology, llm_tool: LLMTool, cache_dir: Optional[pathlib.Path] = None
):
    """Update ontology properties using LLM analysis, only if missing.

    This function uses the LLM tool to analyze and update the properties
    of a given ontology based on its graph content, but only if any key
    property is missing or empty.
    """
    # Only update if any key property is missing or empty
    if not (o.title and o.ontology_id and o.description and o.version):
        props = render_ontology_summary(o.graph, llm_tool, cache_dir=cache_dir)
        o.set_properties(**props.model_dump())


def update_ontology_manager(
    om: OntologyManager, llm_tool: LLMTool, cache_dir: Optional[pathlib.Path] = None
):
    """Update properties for all ontologies in the manager.

    This function iterates through all ontologies in the manager and updates
    their properties using the LLM tool.

    Args:
        om: The ontology manager containing ontologies to update.
        llm_tool: The LLM tool instance for analysis.
        cache_dir: Directory for caching ontology properties.
    """
    # Use tqdm for a nicer progress bar if available.
    try:
        from tqdm import tqdm  # type: ignore

        iterable = tqdm(om.ontologies, desc="Updating ontology properties", unit="onto")
    except Exception:
        # tqdm not installed or failed -> plain iterable
        iterable = om.ontologies

    for o in iterable:
        update_ontology_properties(o, llm_tool, cache_dir=cache_dir)


class ToolBox:
    """A container class for all tools used in the ontology processing workflow.

    This class initializes and manages various tools needed for document processing,
    ontology management, and LLM interactions.

    Args:
        working_directory: Path to the working directory.
        ontology_directory: Optional path to ontology directory.
        model_name: Name of the LLM model to use.
        llm_base_url: Optional base URL for LLM API.
        temperature: Temperature setting for LLM.
        llm_provider: Provider for LLM service (default: "openai").
        neo4j_uri: (optional) URI for Neo4j connection. If provided with neo4j_auth,
                    neo4j will be used as triple store (unless Fuseki is also provided).
        neo4j_auth: (optional) Auth string (user/password) for Neo4j connection.
        fuseki_uri: (optional) URI for Fuseki connection. If provided with fuseki_auth,
                    Fuseki will be used as triple store (preferred over Neo4j).
        fuseki_auth: (optional) Auth string (user/password) for Fuseki connection.
        clean: (optional, default False) If True, triple store (Neo4j or Fuseki) will be initialized as clean (all data deleted on startup).
    """

    def __init__(self, **kwargs):
        working_directory: pathlib.Path = kwargs.pop("working_directory")
        ontology_directory: Optional[pathlib.Path] = kwargs.pop("ontology_directory")
        model_name: str = kwargs.pop("model_name")
        llm_base_url: Optional[str] = kwargs.pop("llm_base_url")
        temperature: float = kwargs.pop("temperature")
        llm_provider: str = kwargs.pop("llm_provider", "openai")
        neo4j_uri: Optional[str] = kwargs.pop("neo4j_uri", None)
        neo4j_auth: Optional[str] = kwargs.pop("neo4j_auth", None)
        fuseki_uri: Optional[str] = kwargs.pop("fuseki_uri", None)
        fuseki_auth: Optional[str] = kwargs.pop("fuseki_auth", None)
        clean: bool = kwargs.pop("clean", False)

        self.llm: LLMTool = LLMTool.create(
            provider=llm_provider,
            model=model_name,
            temperature=temperature,
            base_url=llm_base_url,
        )

        # Filesystem manager for initial ontology loading (if ontology_directory provided)
        self.filesystem_manager: Optional[FilesystemTripleStoreManager] = None
        if ontology_directory:
            self.filesystem_manager = FilesystemTripleStoreManager(
                working_directory=working_directory,
                ontology_path=ontology_directory,
            )

        # Main triple store manager - prefer Fuseki over Neo4j, fallback to filesystem
        if fuseki_uri and fuseki_auth:
            # Extract dataset name from URI if not provided
            dataset = None
            if "/" in fuseki_uri:
                dataset = fuseki_uri.split("/")[-1]
            self.triple_store_manager: TripleStoreManager = FusekiTripleStoreManager(
                uri=fuseki_uri, auth=fuseki_auth, dataset=dataset, clean=clean
            )
        elif neo4j_uri and neo4j_auth:
            self.triple_store_manager: TripleStoreManager = Neo4jTripleStoreManager(
                uri=neo4j_uri, auth=neo4j_auth, clean=clean
            )
        else:
            self.triple_store_manager: TripleStoreManager = (
                FilesystemTripleStoreManager(
                    working_directory=working_directory,
                    ontology_path=ontology_directory,
                )
            )
        self.ontology_manager: OntologyManager = OntologyManager()
        self.converter: ConverterTool = ConverterTool()
        self.chunker: ChunkerTool = ChunkerTool()
        self.aggregator: ChunkRDFGraphAggregator = ChunkRDFGraphAggregator()


def init_toolbox(toolbox: ToolBox, cache_dir: Optional[pathlib.Path] = None):
    """Initialize the toolbox with ontologies and their properties.

    This function fetches ontologies from the triple store and updates
    their properties using the LLM tool. If a filesystem manager is available
    for initial loading, it will be used to load ontologies from files first.

    Args:
        toolbox: The ToolBox instance to initialize.
        cache_dir: Directory for caching ontology properties.
    """
    # If we have a filesystem manager, use it to load initial ontologies
    if toolbox.filesystem_manager:
        initial_ontologies = toolbox.filesystem_manager.fetch_ontologies()
        # Store these ontologies in the main triple store manager
        for ontology in initial_ontologies:
            toolbox.triple_store_manager.serialize_ontology(ontology)

    # Now fetch ontologies from the main triple store manager
    toolbox.ontology_manager.ontologies = (
        toolbox.triple_store_manager.fetch_ontologies()
    )
    update_ontology_manager(
        om=toolbox.ontology_manager, llm_tool=toolbox.llm, cache_dir=cache_dir
    )


def render_ontology_summary(
    graph: RDFGraph,
    llm_tool,
    *,
    max_chars: int = 200_000,
    cache_dir: Optional[pathlib.Path] = None,
) -> OntologyProperties:
    """Generate a summary of ontology properties using the LLM tool.

    The function automatically avoids context length limits and includes caching
    and parallelization for better performance.

    For graphs that serialise to less than ``max_chars`` characters it will
    send the whole Turtle string (behaviour unchanged).

    For larger graphs it will:
    1. Split the serialised Turtle into chunks of at most ``max_chars``
       characters.
    2. Ask the LLM to summarise **each** chunk into ``OntologyProperties`` (parallelized).
    3. Merge the partial results, preferring the first non-empty value for
       every field.
    4. Cache results to avoid re-processing the same ontology.

    Args:
        graph: The RDF graph to analyse.
        llm_tool: The LLM tool instance.
        max_chars: Maximum number of characters allowed per prompt.
        cache_dir: Directory for caching results (defaults to ./cache/ontology_properties).

    Returns:
        OntologyProperties:  Structured ontology metadata.
    """
    # Serialise once (rdflib serialisation is relatively expensive).
    ontology_str = graph.serialize(format="turtle")

    # Set up caching
    if cache_dir is None:
        cache_dir = pathlib.Path("./cache/ontology_properties")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache key based on ontology content hash
    content_hash = hashlib.sha256(ontology_str.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{content_hash}.json"

    # Try to load from cache first
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                return OntologyProperties(**cached_data)
        except Exception:
            # Cache file corrupted, continue with processing
            pass

    # Helper – original single-shot strategy ---------------------------------
    def _single_call(turtle_str: str) -> OntologyProperties:
        parser = DspyOutputParser(pydantic_object=OntologyProperties)
        prompt = PromptTemplate(
            template=(
                "Below is an ontology in Turtle format:\n\n"
                "```ttl\n{ontology_str}\n```\n\n"
                "{format_instructions}"
            ),
            input_variables=["ontology_str"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        response = llm_tool(prompt.format_prompt(ontology_str=turtle_str))
        return parser.parse(response.content)

    # Async helper for parallel chunk processing
    async def _process_chunk_async(
        chunk: str, idx: int, total: int
    ) -> Optional[OntologyProperties]:
        """Process a single chunk asynchronously."""
        parser = DspyOutputParser(pydantic_object=OntologyProperties)
        chunk_prompt = PromptTemplate(
            template=(
                "Below is chunk {idx}/{total} of a large ontology in Turtle format.\n\n"
                "```ttl\n{ontology_str}\n```\n\n"
                "{format_instructions}"
            ),
            input_variables=["ontology_str", "idx", "total"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        try:
            # Run the LLM call in a thread pool since it's synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm_tool(
                    chunk_prompt.format_prompt(
                        ontology_str=chunk, idx=str(idx), total=str(total)
                    )
                ),
            )
            return parser.parse(response.content)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # If small enough – keep previous behaviour
    if len(ontology_str) <= max_chars:
        result = _single_call(ontology_str)
        # Cache the result
        try:
            with open(cache_file, "w") as f:
                json.dump(result.model_dump(), f, indent=2)
        except Exception:
            pass  # Don't fail if caching fails
        return result

    # -----------------------------------------------------------------------
    # Otherwise we have to chunk
    # The chunk size should stay well below model context limits.
    # ``max_chars`` is therefore interpreted as the *absolute* upper bound
    # for characters per prompt (≈ chars/4 ≈ tokens). The default (200 000
    # chars) corresponds to roughly 50 000 tokens, leaving ample room for
    # prompt boilerplate inside a 128 k-token model context.
    chunk_size = max_chars
    chunks: list[str] = [
        ontology_str[i : i + chunk_size]
        for i in range(0, len(ontology_str), chunk_size)
    ]

    total_chunks = len(chunks)

    # Process chunks in parallel using asyncio
    async def _process_all_chunks():
        # Limit concurrency to avoid overwhelming the API
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        # Set up progress tracking
        try:
            from tqdm.asyncio import tqdm_asyncio

            progress_bar = True
        except ImportError:
            progress_bar = False

        async def _process_with_semaphore(chunk: str, idx: int):
            async with semaphore:
                return await _process_chunk_async(chunk, idx, total_chunks)

        tasks = [
            _process_with_semaphore(chunk, idx + 1) for idx, chunk in enumerate(chunks)
        ]

        if progress_bar:
            # tqdm_asyncio.gather doesn't support return_exceptions, so we'll use asyncio.gather
            try:
                return await tqdm_asyncio.gather(
                    *tasks, desc=f"Processing {total_chunks} chunks"
                )
            except Exception:
                # Fallback to regular gather if tqdm fails
                return await asyncio.gather(*tasks, return_exceptions=True)
        else:
            return await asyncio.gather(*tasks, return_exceptions=True)

    # Run the parallel processing
    try:
        # Check if we're already in an async context
        # loop = asyncio.get_running_loop()
        # If we are, we need to create a task instead of using asyncio.run()
        # since asyncio.run() cannot be called from a running event loop
        import concurrent.futures

        # Create a new event loop in a separate thread
        def run_in_new_thread():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_process_all_chunks())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_thread)
            results = future.result()
    except RuntimeError:
        # No running loop, we can run directly
        results = asyncio.run(_process_all_chunks())

    # Filter out None results and exceptions
    partial_props = [r for r in results if isinstance(r, OntologyProperties)]

    # -----------------------------------------------------------------------
    # Merge partial results – take the first non-empty value encountered.
    merged = OntologyProperties()
    for p in partial_props:
        if p.title and not merged.title:
            merged.title = p.title
        if p.ontology_id and not merged.ontology_id:
            merged.ontology_id = p.ontology_id
        if p.description and not merged.description:
            merged.description = p.description
        if p.version and not merged.version:
            merged.version = p.version
        if p.iri and not merged.iri:
            merged.iri = p.iri

    # As a final fallback – if we still have nothing useful – attempt a
    # single call on the *first* ``max_chars`` characters. This additional
    # call is cheap and often contains the ontology header (prefixes, owl:Ontology
    # declaration, etc.).
    if not any([merged.title, merged.ontology_id, merged.description, merged.version]):
        header_sample = ontology_str[:max_chars]
        try:
            merged = _single_call(header_sample)
        except Exception:
            # Give up – return empty object so downstream code can decide
            # what to do.
            pass

    # Cache the final result
    try:
        with open(cache_file, "w") as f:
            json.dump(merged.model_dump(), f, indent=2)
    except Exception:
        pass  # Don't fail if caching fails

    return merged
