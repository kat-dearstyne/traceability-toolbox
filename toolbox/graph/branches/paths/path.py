from dataclasses import dataclass
from typing import Any

from toolbox.graph.branches.conditions.condition import Condition


@dataclass
class Path:
    action: Any
    condition: Condition = None
