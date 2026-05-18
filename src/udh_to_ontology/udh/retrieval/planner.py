import dspy

class QueryPlan(dspy.Signature):
    """
    You are an expert search strategist for a highly structured Knowledge Graph.
    Deconstruct the user_query into optimized target arrays for semantic vector matching.

    CRITICAL CONSTRAINTS:
    1. Precision over Volume: Limit all lists to exactly 1-3 highly precise terms. Outputting long lists dilutes semantic similarity scores.
    2. Strict Separation: 'target_entities' are STRICTLY exact proper nouns/IDs. 'broad_anchors' are STRICTLY generic roles/types. Do NOT overlap them.
    3. Edge Formatting: 'target_edge_labels' MUST be strict UPPERCASE strings, 2-4 words, separated by underscores (e.g., DEFINES_CONDITION, HAS_EXCEPTION, ALTERS_TERM).
    4. Anti-Hallucination: If the query does not explicitly imply a specific field (e.g., no proper nouns are mentioned), output an empty list. Do NOT invent filler data.

    Keep reasoning brief. Once all fields are ready, output immediately and stop.
    """
    user_query: str = dspy.InputField()

    # Node Guidance
    rewritten_query: str = dspy.OutputField(desc="A clean, standalone version of the query for semantic search.")
    target_topics: list[str] = dspy.OutputField(desc="1-3 Broad thematic categories the query touches upon.")
    target_entities: list[str] = dspy.OutputField(desc="1-3 Specific named instances (e.g., 'John Doe', 'Policy 123').")
    broad_anchors: list[str] = dspy.OutputField(desc="1-3Role-based entities or concepts (e.g., 'policyholder', 'claimant', 'vehicle').")

    # Edge Guidance (Updated)
    target_edge_labels: list[str] = dspy.OutputField(desc="1-3 UPPERCASE relationship labels (strict 2-4 word labels) representing desired logical steps (e.g., DEFINES_CONDITION, HAS_EXCEPTION).")
    target_edge_semantics: list[str] = dspy.OutputField(desc="1-3 Rich, descriptive phrases of the narrative relationship needed to connect the dots (e.g., 'exceptions to the liability clause').")

def generate_query_plan(query: str):
    planner = dspy.ChainOfThought(QueryPlan)
    return planner(user_query=query)
