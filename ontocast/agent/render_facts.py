"""Fact rendering agent for OntoCast.

This module provides functionality for rendering facts from RDF graphs into
human-readable formats, making the extracted knowledge more accessible and
understandable.
"""

import logging
import textwrap

from langchain.prompts import PromptTemplate

from ontocast.dspy_output_parser import DspyOutputParser
from ontocast.onto import AgentState, FailureStages, SemanticTriplesFactsReport, Status
from ontocast.prompt.render_facts import (
    ontology_instruction,
)
from ontocast.prompt.render_facts import (
    template_prompt as template_prompt_str,
)
from ontocast.toolbox import ToolBox
from ontocast.util import truncate_ontology_string, truncate_text

logger = logging.getLogger(__name__)


def render_facts(state: AgentState, tools: ToolBox) -> AgentState:
    """Render facts from the current chunk into a human-readable format.

    This function takes the facts in the current chunk and renders them into a
    more accessible format, making the extracted knowledge easier to understand.

    Args:
        state: The current agent state containing the chunk to render.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with rendered facts.
    """
    logger.info("Starting to render facts")
    llm_tool = tools.llm

    parser = DspyOutputParser(pydantic_object=SemanticTriplesFactsReport)

    ontology_str = state.current_ontology.graph.serialize(format="turtle")

    # Truncate ontology string to prevent API limits
    ontology_str = truncate_ontology_string(
        ontology_str, context=truncate_text(state.current_chunk.text)
    )

    ontology_instruction_str = ontology_instruction.format(
        ontology_iri=state.current_ontology.iri, ontology_str=ontology_str
    )

    prompt = PromptTemplate(
        template=template_prompt_str,
        input_variables=[
            "ontology_namespace",
            "current_doc_namespace",
            "text",
            "ontology_instruction",
            "failure_instruction",
            "format_instructions",
        ],
    )

    try:
        if state.status != Status.SUCCESS and state.failure_reason is not None:
            failure_instruction = "The previous attempt to generate triples failed."
            if state.failure_stage is not None:
                failure_instruction += (
                    f"\n\nIt failed at the stage: {state.failure_stage}"
                )
            failure_instruction += f"\n\n{state.failure_reason}"
            failure_instruction += (
                "\n\nPlease fix the errors "
                "and do your best to generate fact triples again."
            )
        else:
            failure_instruction = ""

        # Chunk the input text to create smaller, more manageable prompts
        chunk_texts = textwrap.wrap(
            state.current_chunk.text, 4000, replace_whitespace=False
        )
        reports = []

        for i, sub_chunk_text in enumerate(chunk_texts):
            try:
                response = llm_tool(
                    prompt.format_prompt(
                        ontology_namespace=state.current_ontology.namespace,
                        current_doc_namespace=state.current_chunk.namespace,
                        text=truncate_text(
                            sub_chunk_text
                        ),  # Still truncate sub-chunk just in case
                        ontology_instruction=ontology_instruction_str,
                        failure_instruction=failure_instruction,
                        format_instructions=parser.get_format_instructions(),
                    )
                )
                report = parser.parse(response.content)
                reports.append(report)
            except Exception as e:
                logging.warning(
                    f"Failed to process sub-chunk {i + 1}/{len(chunk_texts)}: {e}"
                )

        # Merge the results from all sub-chunks
        final_report = SemanticTriplesFactsReport()
        for report in reports:
            if report.semantic_graph:
                final_report.semantic_graph += report.semantic_graph

        # Use the average of the scores
        valid_relevance_scores = [
            r.ontology_relevance_score
            for r in reports
            if r.ontology_relevance_score is not None
        ]
        if valid_relevance_scores:
            final_report.ontology_relevance_score = sum(valid_relevance_scores) / len(
                valid_relevance_scores
            )

        valid_generation_scores = [
            r.triples_generation_score
            for r in reports
            if r.triples_generation_score is not None
        ]
        if valid_generation_scores:
            final_report.triples_generation_score = sum(valid_generation_scores) / len(
                valid_generation_scores
            )

        final_report.semantic_graph.sanitize_prefixes_namespaces()
        if state.current_chunk.graph is not None:
            state.current_chunk.graph += final_report.semantic_graph

        state.clear_failure()
        return state

    except Exception as e:
        logger.error(f"Failed to generate triples: {str(e)}")
        state.set_failure(FailureStages.PARSE_TEXT_TO_FACTS_TRIPLES, str(e))
        return state
