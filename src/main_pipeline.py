# main_pipeline.py
import json
import time
from typing import List, Dict, Optional
from atomic_tool import AtomicTool
from tool_library import ToolLibrary
from openai_utils import function_call_completion, atomic_tool_to_openai_tool
from prompt_builder import get_cot_output_main
from tool_selector import select_tool
from new_tool_processor import get_embedding  # 仅用于相似度检查（可选）
from validate_in_docker import run_validate_in_docker
import torch

# 检查是否可计算 embedding（用于相似度去重）
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


def main_pipeline(
    all_datas: List[dict],
    tool_library: ToolLibrary,
    fun_call_api_key: str,
    fun_call_base_url: str,
    cot_output_api_key: str,
    cot_output_base_url: str,
    model_dir: str,
    reranker_dir: str,
    model_path: Optional[str] = None,  # 用于 embedding 相似度检查
    output_file: str = "gpt_tool_100_1119.jsonl"
):
    system_prompt = """You are a strict tool-calling agent. You MUST respond ONLY by invoking one pre-defined AtomicTool.
- NEVER answer directly, compute, or explain in natural language.
- Invoke exactly ONE tool that best matches the user's request.
- If no tool matches, return an empty tool call list ([]).
- Output MUST be valid OpenAI tool call JSON; no extra text allowed."""

    for idx, data in enumerate(all_datas):
        sub_questions = data['sub_question']
        question = data['question']
        print(f"\n[Processing {idx+1}/{len(all_datas)}] Question: {question[:60]}...")
        tool_calls = []

        for i, sub_question in enumerate(sub_questions):
            tool_names, _, tool_embs = tool_library.get_embeddings()
            if tool_embs is not None and len(tool_names) > 0:
                tool_results = select_tool(
                    query=sub_question,
                    model_dir=model_dir,
                    reranker_dir=reranker_dir,
                    tool_library=tool_library,
                    top_k=3
                )
                main_fun_calls = [func[0] for func in tool_results]
                existing_tools = []
                for fun_call in main_fun_calls:
                    tool = tool_library.get_tool(fun_call)
                    if tool:
                        existing_tools.append(atomic_tool_to_openai_tool(tool))

                initial_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sub_question}
                ]
                print("func call processing .........................................")
                fun_call_response = function_call_completion(
                    api_key=fun_call_api_key,
                    base_url=fun_call_base_url,
                    messages=initial_messages,
                    tools=existing_tools,
                    model="gpt-3.5-turbo"
                )
                print("func call done .........................................")
                tool_calls.append({sub_question: fun_call_response})
            else:
                tool_calls.append({sub_question: ""})
        data['fun_call_response'] = tool_calls

        # Build question_and_codes_list (for CoT)
        question_and_codes_list = []
        for fun_call_response in tool_calls:
            sub_question = list(fun_call_response.keys())[0]
            sub_question_codes = fun_call_response[sub_question]
            q_codes = []
            if sub_question_codes:
                for sub_question_code in sub_question_codes:
                    if sub_question_code is None:
                        continue
                    if sub_question_code and sub_question_code.get('tool_calls'):
                        func_names = [tc['function']['name'] for tc in sub_question_code['tool_calls'] if 'function' in tc]
                        if not func_names:
                            continue
                        for func_name in func_names:
                            tool = tool_library.get_tool(func_name)
                            if tool:
                                tool.hit_count += 1
                                tool_library._save_to_file()
                                q_codes.append(tool.code)
            question_and_codes_list.append({sub_question: q_codes})

        # Get CoT output from LLM
        print('Cot Answer Processing ................................................')
        cot_output, code_data_list, answer = get_cot_output_main(
            api_key=cot_output_api_key,
            base_url=cot_output_base_url,
            main_question=question,
            sub_questions_and_code=question_and_codes_list,
            model="gpt-3.5-turbo"
        )
        print('Cot Answer done ................................................')
        data['gpt_cot_output'] = cot_output
        data['gpt_code_data'] = code_data_list
        data['gpt_cot_answer'] = answer

        # >>> NEW LOGIC: Only add tools that PASS validation <<<
        validated_tools = []
        if code_data_list:
            # Step 1: Prepare code for validation
            temp_data_for_validation = {
                "python_tool": [tool["code"] for tool in code_data_list if "code" in tool]
            }

            if temp_data_for_validation["python_tool"]:
                print("🔍 Validating newly generated tools in Docker...")
                validation_results = run_validate_in_docker(
                    temp_data_for_validation,
                    docker_image="python:3.11-slim",
                    timeout=300,
                    max_install_retry=3
                )

                # Attach validation result to each tool
                for i, tool in enumerate(code_data_list):
                    tool['validation_result'] = validation_results[i] if validation_results and i < len(validation_results) else None
                    tool['is_valid'] = bool(
                        tool['validation_result'] and tool['validation_result'].get('success')
                    )
                    validated_tools.append(tool)

                # Save validation summary for output
                if validation_results:
                    data['python_success'] = all(r.get('success', False) for r in validation_results)
                    data['python_stdout'] = '\n---\n'.join(r.get('stdout', '') for r in validation_results)
                    data['python_stderr'] = '\n---\n'.join(r.get('stderr', '') for r in validation_results)
                    data['installed_packages'] = [r.get('installed', []) for r in validation_results]
                else:
                    data['python_success'] = False
                    data['python_stdout'] = ''
                    data['python_stderr'] = 'Validation failed or returned no results'
                    data['installed_packages'] = []
            else:
                data['python_success'] = None
                data['python_stdout'] = ''
                data['python_stderr'] = 'No executable code to validate'
                data['installed_packages'] = []
        else:
            data['python_success'] = None
            data['python_stdout'] = ''
            data['python_stderr'] = 'No new tools generated'
            data['installed_packages'] = []

        # Step 2: ONLY add VALID tools to library (with optional deduplication)
        for tool_dict in validated_tools:
            if not tool_dict.get('is_valid', False):
                print(f"[SKIP] Tool '{tool_dict['name']}' failed validation. Skipping.")
                continue

            required_keys = {'name', 'code', 'text_description', 'io_description', 'test_example'}
            if not required_keys.issubset(tool_dict.keys()):
                print(f"[WARN] Tool '{tool_dict['name']}' has incomplete format. Skipping.")
                continue

            original_name = tool_dict["name"]
            existing_tool = tool_library.get_tool(original_name)

            should_add = True
            if existing_tool is not None:
                if EMBEDDING_AVAILABLE and model_path:
                    # Perform similarity check
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModelForMaskedLM.from_pretrained(model_path)
                        model.eval()
                        embed_new = get_embedding(tool_dict['code'], tokenizer, model)
                        embed_existing = get_embedding(existing_tool.code, tokenizer, model)
                        similarity = torch.nn.functional.cosine_similarity(embed_new, embed_existing).item()
                        print(f"[DEBUG] Similarity with existing tool '{original_name}': {similarity:.4f}")

                        if similarity >= 0.6:
                            print(f"[INFO] Tool '{original_name}' is too similar ({similarity:.2f} >= 0.6). Skipping.")
                            should_add = False
                        else:
                            # Rename to avoid conflict
                            timestamp = int(time.time() * 1000)
                            original_name = f"{original_name}_{timestamp}"
                    except Exception as e:
                        print(f"[WARN] Embedding check failed: {e}. Adding with original name.")

            if should_add:
                new_tool = AtomicTool(
                    name=original_name,
                    code=tool_dict["code"],
                    description=tool_dict["text_description"],
                    input_params=tool_dict['io_description'].get('input', {}),
                    output_params=tool_dict['io_description'].get('output', {}),
                    example=tool_dict['test_example']
                )
                tool_library.add_tool(new_tool)
                print(f"[INFO] ✅ Valid and distinct tool added: {new_tool.name}")
            else:
                print(f"[INFO] ❌ Skipped adding: {original_name}")

        # Save final result
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")