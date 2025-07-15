"""Ontology criticism agent for OntoCast.

This module provides functionality for analyzing and validating ontologies,
ensuring their structural integrity, consistency, and alignment with domain
requirements.
"""

import logging

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from ontocast.onto import (
    ONTOLOGY_NULL_ID,
    AgentState,
    FailureStages,
    OntologyUpdateCritiqueReport,
    Status,
)
from ontocast.prompt.criticise_ontology import prompt_fresh, prompt_update
from ontocast.tool import LLMTool, OntologyManager
from ontocast.toolbox import ToolBox
from ontocast.util import truncate_ontology_string, truncate_text

logger = logging.getLogger(__name__)


def criticise_ontology(state: AgentState, tools: ToolBox) -> AgentState:
    """Analyze and validate the current ontology.

    This function performs a critical analysis of the ontology in the current
    state, checking for structural integrity, consistency, and alignment with
    domain requirements.

    Args:
        state: The current agent state containing the ontology to analyze.
        tools: The toolbox instance providing utility functions.

    Returns:
        AgentState: Updated state with analysis results.
    """
    logger.info("Criticize ontology")
    llm_tool: LLMTool = tools.llm
    om_tool: OntologyManager = tools.ontology_manager
    parser = PydanticOutputParser(pydantic_object=OntologyUpdateCritiqueReport)

    if state.current_chunk is None:
        state.status = Status.FAILED
        return state

    # ---------------------------------------------------------------
    # Prepare context text *once* so it is available in every branch
    # ---------------------------------------------------------------

    chunk_text = truncate_text(state.current_chunk.text)

    # Check if this is a new ontology (either None or ONTOLOGY_NULL_ID)
    is_new_ontology = (
        state.current_ontology.ontology_id is None
        or state.current_ontology.ontology_id == ONTOLOGY_NULL_ID
    )

    if is_new_ontology:
        prompt = prompt_fresh
        ontology_original_str = ""
    else:
        ontology_serialized = state.current_ontology.graph.serialize(format="turtle")

        # Truncate ontology string to prevent API limits (semantic reduction)
        ontology_serialized = truncate_ontology_string(
            # ontology_serialized, context=chunk_text
            ontology_serialized
        )

        ontology_original_str = (
            f"Here is the original ontology:\n```ttl\n{ontology_serialized}\n```"
        )
        prompt = prompt_update

    prompt = PromptTemplate(
        template=prompt,
        input_variables=[
            "ontology_update",
            "document",
            "format_instructions",
            "ontology_original_str",
        ],
    )

    # ------------------------------------------------------------------
    # Also truncate the *ontology update* string using the same chunk context
    # ------------------------------------------------------------------

    ontology_update_serialized = state.ontology_addendum.graph.serialize(
        format="turtle"
    )

    ontology_update_serialized = truncate_ontology_string(
        ontology_update_serialized, context=chunk_text
    )

    response = llm_tool(
        prompt.format_prompt(
            ontology_update=ontology_update_serialized,
            document=chunk_text,
            format_instructions=parser.get_format_instructions(),
            ontology_original_str=ontology_original_str,
        )
    )
    critique: OntologyUpdateCritiqueReport = parser.parse(response.content)
    logger.debug(
        f"Parsed critique report status: {critique.ontology_update_success}, "
        f"score: {critique.ontology_update_score}"
    )

    if is_new_ontology:
        logger.debug("Adding new ontology to manager")
        om_tool.ontologies.append(state.ontology_addendum)
        state.current_ontology = state.ontology_addendum
    else:
        logger.info(f"Updating existing ontology: {state.current_ontology.ontology_id}")
        try:
            om_tool.update_ontology(
                state.current_ontology.ontology_id, state.ontology_addendum.graph
            )
        except ValueError:
            logger.debug(
                f"Ontology {state.current_ontology.ontology_id} not found, treating as new ontology."
            )
            logger.debug("Adding new ontology to manager")
            om_tool.ontologies.append(state.ontology_addendum)
            state.current_ontology = state.ontology_addendum
        except Exception as e:
            logger.error(
                f"Failed to update ontology {state.current_ontology.ontology_id}: {e}"
            )
            state.set_failure(
                stage=FailureStages.ONTOLOGY_CRITIQUE,
                reason=f"Failed to update ontology: {e}",
                success_score=0.0,
            )

    if critique.ontology_update_success:
        logger.info("Ontology critique successful, clearing failure state")
        state.clear_failure()
    else:
        logger.info("Ontology critique failed, setting failure state")
        state.set_failure(
            stage=FailureStages.ONTOLOGY_CRITIQUE,
            reason=critique.ontology_update_critique_comment,
            success_score=critique.ontology_update_score,
        )

    return state
