from typing import Any
from dataclasses import dataclass, field

@dataclass
class AgentState:
    """
    Shared state passed between agents
    
    IMPORTANT: This state is TEMPORARY
    - Created when user sends message
    - Destroyed after response is sent
    - NOT persisted to database
    """
    
    # Input
    question: str
    user_id: str
    
    # Planner output
    plan: str = ""
    sources_needed: list[str] = field(default_factory=list)  # ["documents", "web"]
    confidence: float = 0.0
    
    # Retrieved data
    retrieved_docs: list[dict] = field(default_factory=list)
    web_results: list[dict] = field(default_factory=list)
    
    # Tool results
    tool_results: dict[str, Any] = field(default_factory=dict)
    
    # Critic validation
    is_valid: bool = False
    validation_issues: list[str] = field(default_factory=list)
    
    # Final answer
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    confidence_final: float = 0.0
    
    # Metadata
    search_time_ms: float = 0.0
    error: str = ""