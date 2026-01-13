import time
import json
from typing import List, Optional, Dict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import cosine_similarity

from atomic_tool import AtomicTool
from tool_library import ToolLibrary

# 检查依赖
try:
    import torch
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

def get_embedding(code: str, tokenizer, model, device: str = 'cpu'):
    if not EMBEDDING_AVAILABLE:
        raise RuntimeError("transformers/torch is not installed; unable to compute embeddings.")
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        cls_embedding = hidden_states[:, 0, :]
        return cls_embedding

def process_new_tool(model_path: Optional[str], code_data_list: List[dict], tool_library: ToolLibrary):
    if not code_data_list or not model_path:
        return

    for code_data in code_data_list:
        required_keys = {'name', 'code', 'text_description', 'io_description', 'test_example'}
        if not required_keys.issubset(code_data.keys()):
            print("[WARN] Generated tool format is incomplete; skipping storage.")
            continue

        original_name = code_data["name"]
        existing_tool = tool_library.get_tool(original_name)

        if existing_tool is None:
            new_tool = AtomicTool(
                name=original_name,
                code=code_data["code"],
                description=code_data["text_description"],
                input_params=code_data['io_description'].get('input', {}),
                output_params=code_data['io_description'].get('output', {}),
                example=code_data['test_example']
            )
            tool_library.add_tool(new_tool)
            print(f"[INFO] New tool has been added to the library: {new_tool.name}")
        else:
            if not EMBEDDING_AVAILABLE:
                print("[WARN] transformers is not installed; skipping similarity check.")
                continue

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            model.eval()

            embed_new = get_embedding(code_data['code'], tokenizer, model)
            embed_existing = get_embedding(existing_tool.code, tokenizer, model)
            similarity = cosine_similarity(embed_new, embed_existing).item()
            print(f"[DEBUG] Similarity with existing tool '{original_name}' : {similarity:.4f}")

            if similarity >= 0.6:
                print(f"[INFO] Similarity too high ({similarity:.2f} >= 0.6); considered duplicate, skipping storage.")
            else:
                timestamp = int(time.time() * 1000)
                new_name = f"{original_name}_{timestamp}"
                new_tool = AtomicTool(
                    name=new_name,
                    code=code_data["code"],
                    description=code_data["text_description"],
                    input_params=code_data['io_description'].get('input', {}),
                    output_params=code_data['io_description'].get('output', {}),
                    example=code_data['test_example']
                )
                tool_library.add_tool(new_tool)
                print(f"[INFO] Distinctly different new tool has been added to the library: {new_tool.name} (Similarity={similarity:.2f})")