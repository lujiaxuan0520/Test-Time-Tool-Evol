import time
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIError, RateLimitError

from atomic_tool import AtomicTool

def function_call_completion(
    api_key: str,
    base_url: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict],
    model: str = "gpt-4o-2024-05-13",
    max_tokens: int = 500,
    max_retries: int = 3,
    chunk_size: int = 100
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        print("No tools provided.")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    chunk_tools = [tools[i:i + chunk_size] for i in range(0, len(tools), chunk_size)]
    total_chunks = len(chunk_tools)
    results = []

    for idx, tool_chunk in enumerate(chunk_tools):
        print(f"🧩 Running chunk {idx+1}/{total_chunks} with {len(tool_chunk)} tools...")
        for attempt in range(1, max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_chunk,
                    tool_choice="auto",
                    max_tokens=max_tokens
                )
                time.sleep(5)
                result = response.choices[0].message.model_dump()
                results.append(result)
                print(f"   ✔ Chunk {idx+1} completed.")
                break
            except (RateLimitError, APIError) as e:
                wait_time = 2 ** attempt
                print(f"   ⚠ API error on attempt {attempt}: {e}. Retrying in {wait_time}s...")
                if attempt < max_retries:
                    time.sleep(wait_time)
                else:
                    print("   ❌ All retries failed for this chunk.")
                    return None
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                return None

    print(f"🎉 All chunks completed successfully. Total results = {len(results)}.")
    return results

def atomic_tool_to_openai_tool(atomic_tool: AtomicTool) -> dict:
    properties = {}
    required = []
    input_params = atomic_tool.input_params
    if isinstance(input_params, str):
        pairs = [p.strip() for p in input_params.split(",") if p.strip()]
        parsed_params = {}
        for p in pairs:
            sep = ": " if ": " in p else "：" if "：" in p else None
            if sep:
                name_part, desc_part = p.split(sep, 1)
            elif "（" in p and "）" in p:
                name_part = p.split("（")[0].strip()
                desc_part = p[p.find("（")+1:p.find("）")]
            else:
                name_part, desc_part = p, f"{p} param"
            parsed_params[name_part.strip()] = {"type": "number", "description": desc_part.strip()}
        input_params = parsed_params
    elif not isinstance(input_params, dict):
        input_params = {}

    for param_name, desc in input_params.items():
        param_desc = desc if isinstance(desc, str) else f"{param_name} param"
        properties[param_name] = {"type": "number", "description": param_desc}
        required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": atomic_tool.name,
            "description": atomic_tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }