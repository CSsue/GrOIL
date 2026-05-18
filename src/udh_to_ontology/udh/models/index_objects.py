from typing import List, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel, Field, ConfigDict

class EntityFingerprint(BaseModel):
    # Add this configuration to make instances hashable
    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="The exact surface text of the entity mention as it appears in the chunk."
    )

    type: str = Field(
        ...,
        description="The high-level ontological category (e.g., PERSON, ORGANIZATION, LOCATION, DATE)."
    )

    role: str = Field(
        ...,
        description="A specific, functional 'Micro-Role' (3-8 words) describing what the entity is doing or being in this specific context (e.g., 'policyholder responsible for payment' or 'driver of the red truck'). Do not summarize the whole document."
    )

    relational_anchors: Tuple[str, ...] = Field(
        ...,
        description="A list of specific identifiers, proper nouns, or unique objects locally linked to this entity that act as discriminators (e.g., 'Policy #555', 'Claim 123', 'BMW X5')."
    )

class Attribute(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        description="The exact surface text of the entity mention as it appears in the chunk."
    )

    type: str = Field(
        ...,
        description="The type of the attribute"
    )

    chunk_label: str = Field(
        ...,
        description="The role of the attribute in terms of the chunk."
    )
    
class Fact(BaseModel):
    # Add this configuration to make instances hashable
    model_config = ConfigDict(frozen=True)
    
    name: str
    sentence: str
    macro_topics: List[str]
    chunk_topics: List[str]
    entities: List[EntityFingerprint]

class Chunk(BaseModel):
    # Add this configuration to make instances hashable
    model_config = ConfigDict(frozen=True)

    name: str
    text: str
    summary: str
    topics: List[str]
    facts: List[Fact]