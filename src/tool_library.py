import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple
from collections import defaultdict

from atomic_tool import AtomicTool

# 可选依赖
try:
    import torch
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

class ToolLibrary:
    def __init__(self, max_size: int = 50, save_path: Optional[str] = None, embedder=None):
        self.tools: Dict[str, AtomicTool] = {}
        self.max_size = max_size
        self.save_path = save_path
        self.embedder = embedder
        self.tool_names = []
        self.tool_descriptions = []
        self.tool_embs = None
        self._load_from_file()
        if self.embedder is not None:
            self._update_embeddings()

    def _build_descriptions(self):
        self.tool_names = []
        self.tool_descriptions = []
        for name, tool in self.tools.items():
            desc = tool.description
            if tool.input_params:
                if isinstance(tool.input_params, dict):
                    input_str = ", ".join(f"{k}: {v}" for k, v in tool.input_params.items())
                else:
                    input_str = str(tool.input_params)
                desc += ", 输入: " + input_str
            self.tool_names.append(name)
            self.tool_descriptions.append(desc)

    def _update_embeddings(self):
        if not self.embedder or not self.tools:
            self.tool_embs = None
            return
        self._build_descriptions()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder.to(device)
        try:
            from sentence_transformers import SentenceTransformer
            self.tool_embs = self.embedder.encode(
                self.tool_descriptions,
                normalize_embeddings=True,
                device=device,
                batch_size=8,
                show_progress_bar=False
            )
        finally:
            self.embedder.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def add_tool(self, tool: AtomicTool) -> bool:
        if len(self.tools) >= self.max_size and tool.name not in self.tools:
            self._evict_lowest_hit_tool()
        self.tools[tool.name] = tool
        self._save_to_file()
        if self.embedder is not None:
            self._update_embeddings()
        return True

    def get_tool(self, name: str) -> Optional[AtomicTool]:
        return self.tools.get(name)

    def get_embeddings(self) -> Tuple[list, list, Optional[list]]:
        return self.tool_names, self.tool_descriptions, self.tool_embs

    def _evict_lowest_hit_tool(self) -> None:
        if not self.tools:
            return
        tools_sorted = sorted(
            self.tools.values(),
            key=lambda x: (x.hit_count, x.created_at)
        )
        lowest_tool = tools_sorted[0]
        del self.tools[lowest_tool.name]
        print(f"[INFO] Removed tool: {lowest_tool.name} "
              f"(hits={lowest_tool.hit_count}, created={lowest_tool.created_at})")

    def _save_to_file(self) -> None:
        if not self.save_path:
            return
        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        data = {name: tool.to_dict() for name, tool in self.tools.items()}
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_from_file(self) -> None:
        if not self.save_path or not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, tool_data in data.items():
                    self.tools[name] = AtomicTool.from_dict(tool_data)
        except Exception as e:
            print(f"[ERROR] Failed to load tool library: {e}")