from typing import Optional

from toolbox.data.keys.structure_keys import TraceKeys
from toolbox.util.typed_enum_dict import TypedEnumDict


class Trace(TypedEnumDict, keys=TraceKeys):
    """
    A trace prediction for a pair of artifacts.
    """
    link_id: Optional[int]
    source: str
    target: str
    score: Optional[float]
    label: Optional[int]
    explanation: Optional[str]
    relationship_type: Optional[str]
    color: Optional[str]
