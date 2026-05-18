import dspy
from typing import List, Tuple
from src.dispergraph_to_ontology.dispergraph.models.index_objects import Fact, Chunk
from src.dispergraph_to_ontology.dispergraph.index.extraction.extract_facts import FactAssembler

class ChunkMeta(dspy.Signature):
    __doc__ = """
    Extract structural metadata and core facts from the source_text :
    
    - name: Provide a concise 3-5 word descriptive title for this chunk.
    - topics: Provide a list of 2-4 macro-level thematic keywords or short phrases.
    - summary: Provide a dense, 2-3 sentence summary of the core information.
    - fact_sentences: Extract a list of self-contained factual sentences directly from the text. Ensure each sentence represents a complete clear claim, rule, event, or other set of information.
    
    Once you have all fields ready, output immediately and stop. Keep your reasoning brief. Do not loop. Do not repeat your reasoning. Commit to your answer and output it.
    """

    source_text: str = dspy.InputField()
    topics: list[str] = dspy.OutputField()
    summary: str = dspy.OutputField()
    fact_sentences: list[str] = dspy.OutputField()
    name: str = dspy.OutputField()

class ChunkDescription(dspy.Signature):
    __doc__ = """
    Extract structural metadata and core facts from the source_text :
    
    - name: Provide a concise 3-5 word descriptive title for this chunk.
    - topics: Provide a list of 2-4 macro-level thematic keywords or short phrases.
    - summary: Provide a dense, 3-4 sentence summary of the core information.

    Once you have all fields ready, output immediately and stop. Keep your reasoning brief. Do not loop. Do not repeat your reasoning. Commit to your answer and output it.
    """

    source_text: str = dspy.InputField()
    topics: list[str] = dspy.OutputField()
    summary: str = dspy.OutputField()
    name: str = dspy.OutputField()

class ChunkFacts(dspy.Signature):
    __doc__ = """
    You are an expert intelligence analyst. Your goal is to achieve 100% exhaustive recall of all factual information contained in the source_text.
    
    -Steps-
    1. Scan the text and identify every distinct Entity (person, organization, concept, document, location) and every distinct Event/Action.
    2. For EACH Entity and Event identified, extract the specific, atomic facts associated with them.
    
    Each fact sentence must be:
    * Atomic: It must contain exactly one complete thought, rule, or action. Break compound sentences into multiple separate facts.
    * Self-Contained: It must make perfect sense completely out of context. Resolve all pronouns (change "He signed it" to "John Doe signed Policy 123").
    * Grounded: It must be explicitly stated in the text. Do not infer or summarize external knowledge.
    
    CRITICAL: You must extract EVERYTHING. 
    For example, if a chunk has 15 distinct facts, do not stop after 3 or 4 facts, instead you must output the 15 fact sentences.
    
    Keep your reasoning brief. Once your facts list is complete, output immediately and stop. Do not repeat facts.
    """

    source_text: str = dspy.InputField()
    fact_sentences: list[str] = dspy.OutputField()

class ChunkAssembler(dspy.Module):
    def __init__(self):
        self.chunk_meta_extractor = dspy.ChainOfThought(ChunkMeta)
        self.chunk_description_extractor = dspy.ChainOfThought(ChunkDescription)
        self.chunk_fact_extractor = dspy.ChainOfThought(ChunkFacts)

    def forward(self, source_text: str, double_extraction=True) -> Chunk:
        chunk_description=None
        chunk_meta_info=None
        if double_extraction:
            #chunk_meta_info = self.chunk_meta_extractor(source_text=source_text)
            chunk_description = self.chunk_description_extractor(source_text=source_text)
            chunk_facts = self.chunk_fact_extractor(source_text=source_text)
        else:
            chunk_meta_info = self.chunk_meta_extractor(source_text=source_text)

        fact_assembler = FactAssembler()
        facts = []

        if double_extraction:
            fact_storage = chunk_facts
        else:
            fact_storage = chunk_meta_info

        for fact_sentence in fact_storage.fact_sentences:
            fact = fact_assembler(source_text=source_text, fact_sentence=fact_sentence)
            facts.append(fact)

        if double_extraction:
            chunk_instance = Chunk(
                name=chunk_description.name,
                text=source_text,
                summary=chunk_description.summary,
                topics=chunk_description.topics,
                facts=facts
            )
        else:
            chunk_instance = Chunk(
                name=chunk_meta_info.name,
                text=source_text,
                summary=chunk_meta_info.summary,
                topics=chunk_meta_info.topics,
                facts=facts
            )

        return chunk_instance

def main():
    pass

if __name__ == "__main__":
    main()