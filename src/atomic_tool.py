from datetime import datetime
from typing import Optional, Dict, Any

class AtomicTool:
    def __init__(
        self,
        name: str,
        code: str,
        description: str,
        input_params: dict,
        output_params: dict,
        example: Optional[dict] = None
    ):
        self.name = name
        self.code = code
        self.description = description
        self.input_params = input_params
        self.output_params = output_params
        self.example = example or {}
        self.hit_count = 0
        self.created_at = self._get_timestamp()

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "input_params": self.input_params,
            "output_params": self.output_params,
            "example": self.example,
            "hit_count": self.hit_count,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AtomicTool":
        tool = cls(
            name=data["name"],
            code=data["code"],
            description=data["description"],
            input_params=data["input_params"],
            output_params=data["output_params"],
            example=data.get("example", {})
        )
        tool.hit_count = data.get("hit_count", 0)
        tool.created_at = data.get("created_at", tool._get_timestamp())
        return tool

    def __repr__(self) -> str:
        return f"<AtomicTool name={self.name}, hits={self.hit_count}, created_at={self.created_at}>"