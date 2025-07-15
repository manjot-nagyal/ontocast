"""OntoCast API server implementation.

This module provides a web server implementation for the OntoCast framework
using Robyn. It exposes REST API endpoints for processing documents and
extracting semantic triples with ontology assistance.

The server supports:
- Health check endpoint (/health)
- Service information endpoint (/info)
- Document processing endpoint (/process)
- Multiple input formats (JSON, multipart/form-data)
- Streaming workflow execution
- Comprehensive error handling and logging

The server integrates with the OntoCast workflow graph to process documents
through the complete pipeline: chunking, ontology selection, fact extraction,
and aggregation.

Example:
    python -m ontocast.cli.serve --env-path .env --working-directory ./work
"""

import asyncio
import logging
import logging.config
import os
import pathlib
from typing import Optional

import click
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from robyn import Headers, Request, Response, Robyn, jsonify

from ontocast.cli.util import crawl_directories
from ontocast.onto import AgentState
from ontocast.stategraph import create_agent_graph
from ontocast.toolbox import ToolBox, init_toolbox

logger = logging.getLogger(__name__)


def calculate_recursion_limit(
    max_visits: int, head_chunks: Optional[int] = None
) -> int:
    """Calculate the recursion limit based on max_visits and head_chunks.

    Args:
        max_visits: Maximum number of visits allowed per node
        head_chunks: Optional maximum number of chunks to process

    Returns:
        int: Calculated recursion limit
    """
    base_recursion_limit = int(os.getenv("RECURSION_LIMIT", 1000))
    estimated_chunks = int(os.getenv("ESTIMATED_CHUNKS", 30))
    if head_chunks is not None:
        # If we know the number of chunks, calculate exact limit
        return max(base_recursion_limit, max_visits * head_chunks * 10)
    else:
        # If we don't know chunks, use a conservative estimate
        return max(base_recursion_limit, max_visits * estimated_chunks * 10)


def create_app(tools: ToolBox, head_chunks: Optional[int] = None, max_visits: int = 3):
    app = Robyn(__file__)
    workflow: CompiledStateGraph = create_agent_graph(tools)
    recursion_limit = calculate_recursion_limit(max_visits, head_chunks)

    @app.get("/health")
    async def health_check():
        """MCP health check endpoint."""
        try:
            # Check if LLM is available
            if tools.llm is None:
                return Response(
                    status_code=503,
                    headers=Headers({"Content-Type": "application/json"}),
                    description=jsonify(
                        {"status": "unhealthy", "error": "LLM not initialized"}
                    ),
                )

            return Response(
                status_code=200,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(
                    {
                        "status": "healthy",
                        "version": "0.1.1",
                        "llm_provider": tools.llm_provider,
                    }
                ),
            )
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return Response(
                status_code=503,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify({"status": "unhealthy", "error": str(e)}),
            )

    @app.get("/info")
    async def info():
        """MCP info endpoint."""
        return Response(
            status_code=200,
            headers=Headers({"Content-Type": "application/json"}),
            description=jsonify(
                {
                    "name": "ontocast",
                    "version": "0.1.1",
                    "description": "Agentic ontology assisted framework "
                    "for semantic triple extraction",
                    "capabilities": ["text-to-triples", "ontology-extraction"],
                    "input_types": ["text", "json", "pdf", "markdown"],
                    "output_types": ["turtle", "json"],
                    "llm_provider": tools.llm.provider,
                    "model_name": tools.llm.model,
                }
            ),
        )

    @app.post("/process")
    async def process(request: Request):
        """MCP process endpoint."""
        try:
            content_type = request.headers.get("content-type")
            logger.debug(f"Content-Type: {content_type}")
            logger.debug(f"Request headers: {request.headers}")
            logger.debug(f"Request body: {request.body}")

            if content_type.startswith("application/json"):
                data = request.body
                # Convert string to bytes
                bytes_data = data.encode("utf-8")
                logger.debug(
                    f"Parsed JSON data: {data}, bytes length: {len(bytes_data)}"
                )
                files = {"input.json": bytes_data}
            elif content_type.startswith("multipart/form-data"):
                files = request.files
                logger.debug(f"Files: {files.keys()}")
                logger.debug(f"Files-types: {[(k, type(v)) for k, v in files.items()]}")
                if not files:
                    return Response(
                        status_code=400,
                        headers=Headers({"Content-Type": "application/json"}),
                        description=jsonify(
                            {
                                "status": "error",
                                "error": "No file provided",
                                "error_type": "ValidationError",
                            }
                        ),
                    )
            else:
                logger.debug(f"Unsupported content type: {content_type}")
                return Response(
                    status_code=400,
                    headers=Headers({"Content-Type": "application/json"}),
                    description=jsonify(
                        {
                            "status": "error",
                            "error": f"Unsupported content type: {content_type}",
                            "error_type": "ValidationError",
                        }
                    ),
                )

            state = AgentState(
                files=files, max_visits=max_visits, max_chunks=head_chunks
            )

            async for chunk in workflow.astream(
                state,
                stream_mode="values",
                config=RunnableConfig(recursion_limit=recursion_limit),
            ):
                state = chunk

            # Format response according to MCP specification
            result = {
                "status": "success",
                "data": {
                    "facts": state["aggregated_facts"].serialize(format="turtle"),
                    "ontology": state["current_ontology"].graph.serialize(
                        format="turtle"
                    ),
                },
                "metadata": {
                    "status": state["status"],
                    "chunks_processed": len(state.get("chunks_processed", [])),
                    "chunks_remaining": len(state.get("chunks", [])),
                },
            }

            return Response(
                status_code=200,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(result),
            )

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error("Error traceback:", exc_info=True)
            return Response(
                status_code=500,
                headers=Headers({"Content-Type": "application/json"}),
                description=jsonify(
                    {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_details": {
                            "stage": state.get("failure_stage", "unknown"),
                            "reason": state.get("failure_reason", "unknown"),
                        }
                        if hasattr(state, "failure_stage")
                        else None,
                    }
                ),
            )

    return app


@click.command()
@click.option(
    "--env-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
    default=".env",
    help=(
        "Path to .env file. If NEO4J_URI and NEO4J_AUTH are set, "
        "neo4j will be used as triple store. If FUSEKI_URI and FUSEKI_AUTH are set, "
        "Fuseki will be used as triple store (preferred over Neo4j)."
    ),
)
@click.option(
    "--ontology-directory", type=click.Path(path_type=pathlib.Path), default=None
)
@click.option(
    "--working-directory", type=click.Path(path_type=pathlib.Path), required=True
)
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), default=None)
@click.option("--head-chunks", type=int, default=None)
@click.option(
    "--max-visits",
    type=int,
    default=3,
    help="Maximum number of visits allowed per node",
)
@click.option("--logging-level", type=click.STRING)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="If set, triple store (Neo4j or Fuseki) will be initialized as clean (all data deleted on startup).",
)
def run(
    env_path: pathlib.Path,
    ontology_directory: Optional[pathlib.Path],
    working_directory: pathlib.Path,
    input_path: Optional[pathlib.Path],
    head_chunks: Optional[int],
    max_visits: int,
    logging_level: Optional[str],
    clean: bool,
):
    """
    Main entry point for the OntoCast server/CLI.
    If FUSEKI_URI and FUSEKI_AUTH are set in the environment,
        Fuseki will be used as the triple store backend (preferred).
    If NEO4J_URI and NEO4J_AUTH are set in the environment,
        Neo4j will be used as the triple store backend (if Fuseki not available).
    Otherwise, the filesystem backend is used.

    If --clean is set, the triple store (Neo4j or Fuseki) will be initialized as clean (all data deleted on startup).
    """
    if logging_level is not None:
        try:
            logger_conf = f"logging/logging.{logging_level}.conf"
            logging.config.fileConfig(logger_conf, disable_existing_loggers=False)
            logger.debug("debug is on")
        except Exception as e:
            logger.error(f"could set logging level correctly {e}")

    _ = load_dotenv(dotenv_path=env_path.expanduser())

    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    port = os.getenv("PORT", 8999)

    if llm_provider == "openai" and "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    if working_directory:
        working_directory = working_directory.expanduser()
        working_directory.mkdir(parents=True, exist_ok=True)

    neo4j_uri = os.getenv("NEO4J_URI", None)
    neo4j_auth = os.getenv("NEO4J_AUTH", None)
    fuseki_uri = os.getenv("FUSEKI_URI", None)
    fuseki_auth = os.getenv("FUSEKI_AUTH", None)

    tools: ToolBox = ToolBox(
        llm_provider=llm_provider,
        llm_base_url=os.getenv("LLM_BASE_URL", None),
        model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        temperature=os.getenv("LLM_TEMPERATURE", 0.0),
        working_directory=working_directory,
        ontology_directory=ontology_directory,
        neo4j_uri=neo4j_uri,
        neo4j_auth=neo4j_auth,
        fuseki_uri=fuseki_uri,
        fuseki_auth=fuseki_auth,
        clean=clean,
    )
    # Set up cache directory in the working directory
    cache_dir = working_directory / "cache" / "ontology_properties"
    init_toolbox(tools, cache_dir=cache_dir)

    workflow: CompiledStateGraph = create_agent_graph(tools)

    if input_path:
        input_path = input_path.expanduser()

        files = sorted(
            crawl_directories(
                input_path,
                suffixes=tuple([".json"] + list(tools.converter.supported_extensions)),
            )
        )

        recursion_limit = calculate_recursion_limit(max_visits, head_chunks)

        async def process_files():
            for file_path in files:
                logger.info(f"Processing {file_path.name}")
                try:
                    state = AgentState(
                        files={file_path.as_posix(): file_path.read_bytes()},
                        max_visits=max_visits,
                        max_chunks=head_chunks,
                    )
                    async for _ in workflow.astream(
                        state,
                        stream_mode="values",
                        config=RunnableConfig(recursion_limit=recursion_limit),
                    ):
                        pass

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

        asyncio.run(process_files())
    else:
        app = create_app(tools, head_chunks, max_visits=max_visits)
        logger.info(f"Starting MCP-ready server on port {port}")
        app.start(port=port)


if __name__ == "__main__":
    run()
