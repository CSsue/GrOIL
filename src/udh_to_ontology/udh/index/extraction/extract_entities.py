from typing import List, Tuple
from src.dispergraph_to_ontology.dispergraph.models.index_objects import EntityFingerprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy

class TextEntities(dspy.Signature):
    __doc__ = """
    Analyze the provided source_text and extract a comprehensive, deduplicated list of all key entities. 
    Entities include people, organizations, locations, specialized concepts, dates, and other unique identifiers.
    CRITICAL: Extract the best name for each entity. The best name may either be the surface text or you may need to normalize the name.
    Once your list is complete, output immediately and stop. Keep reasoning brief. Do not repeat entities.
    """
    
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")

class TextEntityFingerprints(dspy.Signature):
    __doc__ = """
    Analyze the source_text to identify all key entities and construct a contextual fingerprint for each.
    For every entity found, determine:
    1. name: The exact surface text from the source.
    2. type: The high-level ontological category (e.g., PERSON, ORGANIZATION, LOCATION, CONCEPT).
    3. role: A 3-8 word 'Micro-Role' explicitly defining what the entity is doing or its function in this exact text.
    4. relational_anchors: A list of specific identifiers (e.g., policy numbers, specific car models, dates) locally linked to this entity.
    Do not hallucinate external facts. If an entity has no relational anchors, return an empty list for that field.
    CRITICAL: Output fields must be named EXACTLY as specified. Example:
        - "relational_anchors" (NOT "relational_anks", NOT "relational_anchor")
    Once your list is complete, output immediately and stop. Keep reasoning brief. Do not repeat.
    """
    
    source_text: str = dspy.InputField()
    fingerprints: list[EntityFingerprint] = dspy.OutputField(desc="THOROUGH list of key entities with their contextual fingerprints")
    
class ExtractEntityFingerprintSingle(dspy.Signature):
    __doc__ = """
    Given the source_text and a specific target_entity, generate a precise contextual fingerprint for that entity ONLY.
    Determine the entity's high-level ontological 'type'. 
    Extract a concise 3-8 word 'Micro-Role' that defines its exact function or action within this specific context. 
    Identify any 'relational_anchors' (unique identifiers, proper nouns, or specific numeric IDs) directly associated with it.
    Base the fingerprint STRICTLY on the provided text.
    CRITICAL: Output fields must be named EXACTLY as specified. Example:
        - "relational_anchors" (NOT "relational_anks", NOT "relational_anchor")
    Once you have the fingerprint ready, output immediately and stop.
    """
    
    source_text: str = dspy.InputField()
    target_entity: str = dspy.InputField(desc="The specific entity mention to extract a fingerprint for.")
    fingerprint: EntityFingerprint = dspy.OutputField(desc="The contextual fingerprint of the target entity")

class ExtractEntityFingerprintBatch(dspy.Signature):
    __doc__ = """
    Given the source_text and a list of target_entity names, generate a contextual fingerprint for EACH entity in the list.
    CRITICAL INSTRUCTION: You MUST return a list of fingerprints that perfectly matches the length and order of the input target_entity list.
    For each entity, determine its ontological 'type', a 3-8 word contextual 'Micro-Role', and a list of directly linked 'relational_anchors'. 
    Do not skip any entities. If information is sparse for a specific entity, provide the best local context available without hallucinating.
    CRITICAL: Output fields must be named EXACTLY as specified. Example:
        - "relational_anchors" (NOT "relational_anks", NOT "relational_anchor")
    Once you have a fingerprint for every entity in the list, output immediately and stop. Do not repeat or re-analyze.
    """

    source_text: str = dspy.InputField()
    target_entity: list[str] = dspy.InputField(desc="List of entities to fingerprint.")
    fingerprints: list[EntityFingerprint] = dspy.OutputField(desc="The contextual fingerprint of the target entity")

#TODO: Possibly consider making this process each entity 1 by 1 since its already in parallel and is not a botteneck
# We can add additional parameters to control this.
def process_entities_batch(source_text: str, entities: list[str]):
    batch_fingerprint_generator = dspy.ChainOfThought(ExtractEntityFingerprintBatch)
    return batch_fingerprint_generator(source_text=source_text, entities=entities).fingerprints

def extract_entity_fingerprints(text: str, mode="single_step", batched=True) -> List[EntityFingerprint]:
    if mode not in ["single_step", "multi_step"]:
        raise ValueError(f"Invalid mode in function extract_entities. You chose: {mode}")

    if mode == "single_step":
        generator = dspy.ChainOfThought(TextEntityFingerprints)
        return generator(source_text=text).fingerprints
    elif mode == "multi_step":
        generator = dspy.ChainOfThought(TextEntities)
        entities_str = generator(source_text=text).entities
        if not batched:
            fingerprints = []
            for entity in entities_str:
                entity_fingerprint_extractor = dspy.ChainOfThought(ExtractEntityFingerprintSingle)
                fingerprint = entity_fingerprint_extractor(source_text=text,target_entity=entity).fingerprint
                fingerprints.append(fingerprint)
            return fingerprints
        elif batched:
            fingerprints = []
            batches = [entities_str[i:i+3] for i in range(0, len(entities_str), 3)]
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(
                        process_entities_batch,
                        text,
                        entities_group
                    ): i
                    for i, entities_group in enumerate(batches)
                }

                for future in as_completed(futures):
                    group_fingerprints = future.result()
                    fingerprints.extend(group_fingerprints)
                return fingerprints

def main():
    pass

if __name__ == "__main__":
    main()
    