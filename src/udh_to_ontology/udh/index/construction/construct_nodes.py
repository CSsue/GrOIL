from src.dispergraph_to_ontology.dispergraph.models.Graph import Graph
from src.dispergraph_to_ontology.dispergraph.models.index_objects import *
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
import dspy
from src.dispergraph_to_ontology.dispergraph.utils.chunking import *
from src.dispergraph_to_ontology.dispergraph.index.extraction.extract_chunk_features import ChunkAssembler

def initialize_graph_from_text(text:str, chunk_size=600, overlap=50) -> Graph:
    """
    Takes a text string, divides into chunks, and creates a graph from the chunks

    This graph doesn't include the following higher order edges such as edges between:
    chunks to chunks,
    facts to other facts,
    entities to facts.
    """

    chunks = chunk_text_by_tokens(text=text, chunk_size=chunk_size, overlap=overlap)
    graph = Graph()

    chunk_assembler = ChunkAssembler()
    for chunk in chunks:
        chunk_instance = chunk_assembler(source_text=chunk)
        graph.add_chunk(chunk_instance)

    print("Initializing graph...")
    return graph





