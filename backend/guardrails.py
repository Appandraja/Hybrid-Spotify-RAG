import re
from pydantic import BaseModel, Field, field_validator

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+all\s+previous\s+instructions",
    r"ignore\s+above\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+an?\s+unrestricted",
    r"bypass\s+safety",
    r"jailbreak",
    r"override\s+rules",
]

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="The user question")
    top_k: int = Field(default=4, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def validate_safety(cls, value: str) -> str:
        clean_text = value.strip()
        if not clean_text:
            raise ValueError("Query cannot be empty or blank.")
        
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, clean_text, re.IGNORECASE):
                raise ValueError("Security Alert: Query contains potential prompt-injection patterns.")
        
        return clean_text

class QueryResponse(BaseModel):
    query: str
    answer: str
    vector_evidence: list[dict]
    graph_evidence: list[dict]
    guardrail_passed: bool = True

class EvaluationRequest(BaseModel):
    test_queries: list[str] = Field(default_factory=lambda: [
        "What microservices manage user profiles and auth?",
        "What database is used for analytics and audio events?",
        "How is content ingested by creators and processed?"
    ])