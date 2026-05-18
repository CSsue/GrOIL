from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

from src.dispergraph_to_ontology.dispergraph.models.index_objects import Fact
from src.dispergraph_to_ontology.dispergraph.index.extraction.extract_entities import extract_entity_fingerprints

class FactMetaExtractor(dspy.Signature):
    __doc__ = """
    Analyze the specific fact_sentence within the broader context of its parent source_text.
    Extract precise structural metadata using the following STRICT CONSTRAINTS:
    
    1. name: Generate a concise, descriptive 3-5 word title summarizing the core claim, rule, or event of the fact.
    2. chunk_topics: Extract a list of 2-4 highly specific thematic keywords or short phrases that connect this fact to the broader text.
    
    CRITICAL INSTRUCTIONS:
    - You MUST output at least one valid string in the chunk_topics list.
    - Do NOT return an empty list.
    - Do NOT hallucinate concepts, external facts, or metadata not explicitly supported by the provided text.
    - Keep your reasoning to 3-4 sentences MAXIMUM. Do not over-explain.
    - If you are uncertain about any field, make your best guess and OUTPUT IMMEDIATELY. Do not loop or re-analyze.
    - Once you have a valid name and at least one topic, STOP and return your answer.
    """

    source_text: str = dspy.InputField()
    fact_sentence: str = dspy.InputField()
    name: str = dspy.OutputField()
    chunk_topics: List[str] = dspy.OutputField()

class FactAssembler(dspy.Module):
    def __init__(self):
        self.fact_meta_extractor = dspy.ChainOfThought(FactMetaExtractor)

    def forward(self, source_text: str, fact_sentence: str) -> Fact:
        fact_meta_info = self.fact_meta_extractor(source_text=source_text, fact_sentence=fact_sentence)
        entity_fingerprints = extract_entity_fingerprints(text=source_text)

        fact_instance = Fact(
            name=fact_meta_info.name,
            sentence=fact_sentence,
            macro_topics=[],
            chunk_topics=fact_meta_info.chunk_topics,
            entities=entity_fingerprints,
        )

        return fact_instance

def main():
    pass

if __name__ == "__main__":
    main()
