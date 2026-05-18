import os
import json
import uuid
import networkx as nx
from cudnn import Graph
from pydantic import BaseModel
from src.dispergraph_to_ontology.dispergraph.models.index_objects import EntityFingerprint, Chunk, Fact

class Graph:
    network = nx.DiGraph()
    entities = {}
    chunks = {}
    facts = {}

    def __init__(self):
        # Everything initialized HERE belongs uniquely to the specific instance
        self.network = nx.DiGraph()
        self.entities = {}
        self.chunks = {}  # Added missing initialization
        self.facts = {}   # Added missing initialization

    def set_graph(self, net: nx.DiGraph):
        self.network = net
        self.entities = {}
        self.chunks = {}  # Make sure to clear these on reset too!
        self.facts = {}  # Make sure to clear these on reset too!

        # Iterate over all nodes and save the custom python objects into the dictionaries
        for node, attrs in net.nodes(data=True):
            if attrs.get("type") == "entity":
                data_obj = attrs.get("data")
                if data_obj is not None:
                    self.entities[node] = data_obj
            if attrs.get("type") == "chunk":
                data_obj = attrs.get("data")
                if data_obj is not None:
                    self.chunks[node] = data_obj
            if attrs.get("type") == "fact":
                data_obj = attrs.get("data")
                if data_obj is not None:
                    self.facts[node] = data_obj

    # graph modification functions
    def add_entities(self, entities: list[dict]):
        ids = []
        for entity in entities:
            id = str(uuid.uuid4())
            while self.has_entity(id):
                id = str(uuid.uuid4())
            self.entities[id] = entity
            self.network.add_node(id, data=entity, type="entity")
            ids.append(id)
        return ids

    def add_entity(self, entity):
        id = str(uuid.uuid4())
        while self.has_entity(id):
            id = str(uuid.uuid4())
        self.entities[id] = entity
        self.network.add_node(id, data=entity, type="entity")
        return id

    def add_fact(self, fact, cascading=True):
        id = str(uuid.uuid4())
        while self.has_node_id(id):
            id = str(uuid.uuid4())

        # add fact to storage
        self.facts[id] = fact
        self.network.add_node(id, data=fact, type="fact")

        # if we are not cascading, then we don't add the entities that are nested into the fact.
        if not cascading:
            return id

        entity_ids = self.add_entities(fact.entities)
        for entity_id in entity_ids:
            entity_type = self.entities[entity_id].type
            self.network.add_edge(id, entity_id, edge_type="fact_entity", label=entity_type)
        return id

    def add_chunk(self, chunk, cascading=True):
        id = str(uuid.uuid4())
        while self.has_node_id(id):
            id = str(uuid.uuid4())

        self.chunks[id] = chunk
        self.network.add_node(id, data=chunk, type="chunk")

        # don't go into facts and entities just stop here if not cascading.
        if not cascading:
            return id

        for fact in chunk.facts:
            fact_id = self.add_fact(fact)
            self.network.add_edge(id, fact_id, edge_type="chunk_fact")

        return id

    def merge(self, graph: 'Graph'):
        """
        This function takes a graph and merges it into the current graph.
        It avoids ID collisions by generating new IDs for all incoming nodes,
        and uses an ID mapping dictionary to perfectly reconstruct the edges.
        """
        # Dictionary mapping old IDs from the incoming graph to new IDs in this graph
        id_mapping = {}

        # 1. Iterate through all nodes in the incoming graph
        for old_id, attrs in graph.network.nodes(data=True):
            node_type = attrs.get("type")
            data_obj = attrs.get("data")

            if data_obj is None:
                continue

            # Add each node according to its type, ensuring cascading is False
            # so we can manually control the ID mapping and edge reconstruction later.
            if node_type == "chunk":
                new_id = self.add_chunk(data_obj, cascading=False)
                id_mapping[old_id] = new_id

            elif node_type == "fact":
                new_id = self.add_fact(data_obj, cascading=False)
                id_mapping[old_id] = new_id

            elif node_type == "entity":
                # Entities don't have a cascading parameter as they are the lowest level
                new_id = self.add_entity(data_obj)
                id_mapping[old_id] = new_id

        # 2. Iterate through all the edges in the incoming graph
        # Passing data=True ensures we retrieve the dictionary of edge attributes
        for u, v, edge_data in graph.network.edges(data=True):
            # Look up the new IDs corresponding to the old edge connections
            new_u = id_mapping.get(u)
            new_v = id_mapping.get(v)

            # Ensure both nodes were successfully mapped (safety check)
            if new_u is not None and new_v is not None:
                # Add the edge to the current graph.
                # Unpacking **edge_data ensures all metadata (label, description, score, etc.)
                # is kept exactly the same.
                self.network.add_edge(new_u, new_v, **edge_data)

    def label_edges(self):
        """
        Iterates through all edges in the graph, determines the type of the
        source and target nodes by checking the storage dictionaries, and
        repairs/labels the edge's 'edge_type' attribute accordingly.
        """
        for u, v in self.network.edges():
            # Determine source node type
            if u in self.chunks:
                source_type = "chunk"
            elif u in self.facts:
                source_type = "fact"
            elif u in self.entities:
                source_type = "entity"
            else:
                source_type = "unknown"

            # Determine target node type
            if v in self.chunks:
                target_type = "chunk"
            elif v in self.facts:
                target_type = "fact"
            elif v in self.entities:
                target_type = "entity"
            else:
                target_type = "unknown"

            # Construct the proper edge label and apply it to the edge
            repaired_edge_type = f"{source_type}_{target_type}"
            self.network[u][v]["edge_type"] = repaired_edge_type

    # # graph load and export functions
    # def load_graph(self, path: str):
    #     # 1. Read the graphml file into a new NetworkX DiGraph
    #     net = nx.read_graphml(path)

    #     # 2. Reconstruct the Pydantic objects from the JSON strings
    #     for node, attrs in net.nodes(data=True):
    #         # We only care about nodes that have a 'data' attribute stored as a string
    #         if "data" in attrs and isinstance(attrs["data"], str):
    #             node_type = attrs.get("type")
    #             json_str = attrs["data"]

    #             # Use Pydantic's built-in model_validate_json to reconstruct the object
    #             if node_type == "entity":
    #                 attrs["data"] = EntityFingerprint.model_validate_json(json_str)
    #             elif node_type == "chunk":
    #                 attrs["data"] = Chunk.model_validate_json(json_str)
    #             elif node_type == "fact":
    #                 attrs["data"] = Fact.model_validate_json(json_str)

    #     # 3. Feed the fully restored graph into your existing set_graph method
    #     self.set_graph(net)

    # def export_graph(self, path: str):
    #     H = self.network.copy()

    #     for node, attrs in H.nodes(data=True):
    #         if "data" in attrs and isinstance(attrs["data"], BaseModel):
    #             attrs["data"] = attrs["data"].model_dump_json()

    #     nx.write_graphml(H, path)


    def export_graph(self, path: str):
        H = self.network.copy()

        for node, attrs in H.nodes(data=True):
            if "data" in attrs and isinstance(attrs["data"], BaseModel):
                attrs["data"] = attrs["data"].model_dump_json()

        nx.write_graphml(H, path)

        sidecar_path = path.replace(".graphml", ".json")
        with open(sidecar_path, "w") as file:
            json.dump({
                "entities": {k: v.model_dump() for k, v in self.entities.items()},
                "chunks":   {k: v.model_dump() for k, v in self.chunks.items()},
                "facts":    {k: v.model_dump() for k, v in self.facts.items()},
            }, file)

    def load_graph(self, path: str):
        net = nx.read_graphml(path)

        for node, attrs in net.nodes(data=True):
            if "data" in attrs and isinstance(attrs["data"], str):
                node_type = attrs.get("type")
                json_str = attrs["data"]
                if node_type == "entity":
                    attrs["data"] = EntityFingerprint.model_validate_json(json_str)
                elif node_type == "chunk":
                    attrs["data"] = Chunk.model_validate_json(json_str)
                elif node_type == "fact":
                    attrs["data"] = Fact.model_validate_json(json_str)

        self.set_graph(net)

        sidecar_path = path.replace(".graphml", ".json")
        if os.path.exists(sidecar_path):
            with open(sidecar_path, "r") as file:
                sidecar = json.load(file)
            self.entities = {k: EntityFingerprint.model_validate(v) for k, v in sidecar["entities"].items()}
            self.chunks   = {k: Chunk.model_validate(v) for k, v in sidecar["chunks"].items()}
            self.facts    = {k: Fact.model_validate(v) for k, v in sidecar["facts"].items()}

            # Patch data back into network nodes so merge() can see them
            for node_id, entity in self.entities.items():
                if node_id in self.network.nodes:
                    self.network.nodes[node_id]["data"] = entity
            for node_id, chunk in self.chunks.items():
                if node_id in self.network.nodes:
                    self.network.nodes[node_id]["data"] = chunk
            for node_id, fact in self.facts.items():
                if node_id in self.network.nodes:
                    self.network.nodes[node_id]["data"] = fact
                    
    # graph utility functions
    def has_entity(self, entity_id: str):
        return entity_id in self.entities

    def has_node_id(self, node_id: str):
        return node_id in self.network.nodes

    def get_chunk_ids(self):
        return list(self.chunks.keys())

    def get_entity_ids(self):
        return list(self.entities.keys())

    def has_chunk_chunk_edge(self, id1: str, id2: str):
        return self.network.has_edge(id1, id2)

    def avg_entity_edges(self) -> float:
        num_entities = len(self.entities)

        # Safety check to prevent division by zero if the graph has no entities yet
        if num_entities == 0:
            return 0.0

        # self.network.out_degree(node_id) returns the number of outgoing edges for that node.
        # We sum these up for every ID currently stored in self.entities.
        total_entity_edges = sum(self.network.out_degree(entity_id) for entity_id in self.entities)

        return total_entity_edges / num_entities


    def get_fact_ids(self, by_chunk_id: str = None, mode="list"):
        if by_chunk_id is not None:
            # Return an empty list if the chunk doesn't exist
            if by_chunk_id not in self.network:
                return []

            fact_ids = []
            # Iterate through all outgoing edges from this specific chunk
            for _, target_node, edge_data in self.network.out_edges(by_chunk_id, data=True):
                # Filter strictly for chunk-to-fact relationships
                if edge_data.get("edge_type") == "chunk_fact":
                    fact_ids.append(target_node)

        else:
            # If no chunk_id is provided, return all fact IDs
            fact_ids = list(self.facts.keys())

        if mode == "dict":
            dict_ids = {}
            for fact_id in fact_ids:
                dict_ids[fact_id] = self.facts[fact_id].sentence

            return dict_ids

        return fact_ids





