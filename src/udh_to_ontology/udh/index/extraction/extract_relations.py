from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from src.dispergraph_to_ontology.dispergraph.models.index_objects import Chunk, Fact
from src.dispergraph_to_ontology.dispergraph.index.extraction.extract_entities import extract_entity_fingerprints
from pydantic import BaseModel, Field

class ChunkEdge(dspy.Signature):
    __doc__ = """
    Evaluate the narrative or logical relationship between chunk_summary_1 and chunk_summary_2.
    Provide bidirectional analysis using the following constraints:
    - description_1_2: Explain how chunk 2 builds upon, contrasts, or relates to chunk 1 from chunk 1's perspective (1-2 sentences).
    - description_2_1: Explain how chunk 1 builds upon, contrasts, or relates to chunk 2 from chunk 2's perspective (1-2 sentences).
    - label_1_2 & label_2_1: Provide a strict 2-4 word UPPERCASE relationship label for each direction (e.g., PROVIDES_EXAMPLE, CONTRADICTS, ELABORATES_ON, PRECEDES).
    - relevance_score: Provide a float between 0.0 and 1.0 indicating how strongly these chunks are logically connected (1.0 = highly connected, 0.0 = unrelated).
    
    Keep reasoning brief. Once all fields are ready, output immediately and stop.
    """

    chunk_summary_1: str = dspy.InputField()
    chunk_summary_2: str = dspy.InputField()

    # Replaced single description with directed descriptions
    description_1_2: str = dspy.OutputField(
        desc="How chunk 2 builds upon, contrasts, or relates to chunk 1 from chunk 1's perspective.")
    description_2_1: str = dspy.OutputField(
        desc="How chunk 1 builds upon, contrasts, or relates to chunk 2 from chunk 2's perspective.")

    label_1_2: str = dspy.OutputField()
    label_2_1: str = dspy.OutputField()

    relevance_score: float = dspy.OutputField()

class ChunkFactEdge(dspy.Signature):
    __doc__ = """
    Determine the specific functional role that the provided fact_sentence plays within the broader context of its parent chunk_summary.
    Output MUST be a strict 2-4 word UPPERCASE string describing this structural relationship (e.g., SUPPORTS_CLAIM, PROVIDES_METRIC, DEFINES_TERM, LISTS_EXCLUSION, ETC.).
    Keep reasoning brief. Once the label is ready, output immediately and stop.
    """

    chunk_summary: str = dspy.InputField()
    fact_sentence: str = dspy.InputField()
    short_edge_label: str = dspy.OutputField()

class CandidateFacts(dspy.Signature):
    __doc__ = """
    You are a strict relevance filter. You are given two chunk summaries, a description of their relationship (rich_edge_description), and a fact_dict mapping UUIDs to Fact Sentences.
    Your task is to identify which specific facts from the fact_dict directly participate in or support the described relationship between the two chunks.
    CRITICAL INSTRUCTION: The output candidate_facts MUST be a list containing ONLY the exact dictionary keys (the UUID strings) of the relevant facts. Do NOT output the text of the sentences. Do not hallucinate keys.
    Keep reasoning brief. Once your list is ready, output immediately and stop.
    """

    chunk_summary_1: str = dspy.InputField()
    chunk_summary_2: str = dspy.InputField()
    rich_edge_description: str = dspy.InputField()
    fact_dict: dict = dspy.InputField()
    candidate_facts: list[str] = dspy.OutputField()

class FactEdge(dspy.Signature):
    __doc__ = """
    Analyze the cross-chunk logical relationship between fact_sentence_1 and fact_sentence_2, using their respective chunk summaries for background context.
    - description_1_2 & description_2_1: Provide concise (1-2 sentence) bidirectional descriptions of how the facts interact, complement, or contradict each other.
    - label_1_2 & label_2_1: Provide strict 2-4 word UPPERCASE relationship labels (e.g., CORROBORATES, DEPENDS_ON, CONFLICTS_WITH).
    - relevance_score: A float between 0.0 and 1.0 indicating relationship strength. Output 0.0 if they are completely independent.

    Keep reasoning brief. Once all fields are ready, output immediately and stop.
    """
    # inputs
    fact_sentence_1: str = dspy.InputField()
    chunk_summary_1: str = dspy.InputField()
    fact_sentence_2: str = dspy.InputField()
    chunk_summary_2: str = dspy.InputField()

    # outputs
    description_1_2: str = dspy.OutputField()
    description_2_1: str = dspy.OutputField()
    label_1_2: str = dspy.OutputField()
    label_2_1: str = dspy.OutputField()
    relevance_score: float = dspy.OutputField()

class SameChunkFactEdge(dspy.Signature):
    __doc__ = """
    Analyze the intra-chunk relationship between fact_sentence_1 and fact_sentence_2, which both belong to the provided shared chunk_summary.
    - description_1_2 & description_2_1: Provide concise (1-2 sentence) bidirectional descriptions explaining the local flow of logic between these two facts.
    - label_1_2 & label_2_1: Provide strict 2-4 word UPPERCASE relationship labels (e.g., CAUSES, PRECEDES, EXEMPLIFIES, MODIFIES).
    - relevance_score: A float between 0.0 and 1.0 indicating their degree of logical dependency.
    If there is no relationship between fact sentence 1 and fact sentence 2, you can return empty strings for the descriptions and labels with a relevance score of 0.
    Keep reasoning brief. Once all fields are ready, output immediately and stop.
    """
    # inputs
    chunk_summary: str = dspy.InputField()
    fact_sentence_1: str = dspy.InputField()
    fact_sentence_2: str = dspy.InputField()

    # outputs
    description_1_2: str = dspy.OutputField()
    description_2_1: str = dspy.OutputField()
    label_1_2: str = dspy.OutputField()
    label_2_1: str = dspy.OutputField()
    relevance_score: float = dspy.OutputField()

class EntityFactRelationship(BaseModel):
    entity_name: str = Field(
        ...,
        description="The entity name of the entity whose relationship is to be analyzed."
    )
    relation_name: str = Field(
        ...,
        description="The relationship between the entity and the fact or event."
    )

class EntityFactEdge(dspy.Signature):
    ___doc__ = """
    Analyze how the specific entities in the EntityNames list participate in the provided fact_sentence.
    For EACH entity in the list, generate an EntityFactRelationship object.
    CRITICAL CONSTRAINTS:
    1. entity_name: This MUST exactly match the string provided in the EntityNames input array. Do not alter casing or spelling.
    2. relation_name: Provide a strict 2-4 word UPPERCASE label defining the entity's precise action, state, or role within this specific fact (e.g., INITIATED_ACTION, SUFFERED_LOSS, IS_SUBJECT_OF, LOCATED_AT).
    You must output a relationship for every entity provided in the list.

    Keep reasoning brief. Once you have a relationship for every entity, output immediately and stop.
    """

    #inputs
    fact_name: str = dspy.InputField()
    fact_sentence: str = dspy.InputField()
    EntityNames: list[str] = dspy.InputField()

    #outputs
    relations: list[EntityFactRelationship] = dspy.OutputField()



def main():
    pass

if __name__ == "__main__":
    main()
