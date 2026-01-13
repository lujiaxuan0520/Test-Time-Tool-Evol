import re
import json
import time
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

def build_smart_prompt_main(
    main_question: str,
    sub_questions_and_code: List[Dict[str, List[str]]]
) -> str:
    prompt = f"""You are an intelligent team composed of experts from multiple disciplines, equipped with interdisciplinary comprehensive analysis and computational capabilities.  
# Please strictly combine each sub-question and its corresponding code to answer the main question.

# Input

main question:
{main_question}

List of sub-questions (to be processed in order). Each sub-question is provided as a dictionary; if its value is an empty string, it is treated as "none"：
"""
    for i, item in enumerate(sub_questions_and_code, 1):
        sub_question = list(item.keys())[0]
        code = item[sub_question]
        display_code = code if code else "null"
        prompt += f"\n## step {i}: {sub_question}\ncode: {display_code}\n"

    prompt += """
# Enforcement Rules
1. Execution Strategy
   - If the sub-question provides non-empty and valid code, **simulate its execution** and output a structured process and result.  
   - If the sub-question's code is `"none"` or the provided code cannot correctly solve the sub-problem, generate a runnable Python function (using snake_case naming, including a docstring and I/O description), and place it inside a JSON object within `<code>`, then simulate its execution.
2. Numerical Values and Units
   - Floating-point numbers: Use scientific notation with **6 significant digits**.
   - Probabilities / Ratios (in range 0–1): Round to **4 decimal places**.
   - Physical quantities: Must include **SI units**. If units are missing, explicitly state the **assumed default units** in the JSON.

3. Output Format (Strictly Enforced)
   - **Part 1**：For sub-question with `"none"` code only, generate a **valid JSON object** (parsable by `json.loads()`) as specified.
     <code>
     [
       {
         "sub_question": "the description of question",
         "name": "function name",
         "code": "complete Python function (string)",
         "text_description": "brief function description ",
         "io_description": {
           "input": "parameter name: description, ...",
           "output": "parameter name: description"
         },
         "test_example": {
           "input": input value,
           "result": "result (scientific notation + unit or null)"
         }
       }
     ]
     </code>
     - When calculation is impossible, set `result` to `null`，and optionally include an `error` field with a brief reason.
     - For sub-question that already provide code, do not generate a new function.

   - **Part 2**：The final answer to the main question must be placed within：
     ```
     <answer>
     Plain natural language conclusion (without terms like "code" or "function")
     </answer>
     ```
4. Conciseness Requirements
   - Aside from `<code>` and `<answer>` ，**no extra text is allowed.**。
   - The JSON must be syntactically valid (use double quotes, correct commas, etc.).
# End
Please strictly follow the above format for your response.
"""
    return prompt

def get_cot_output_main(
    api_key,
    base_url,
    main_question: str,
    sub_questions_and_code: List[Dict[str, str]],
    model: str = 'gpt-4o-2024-05-13',
    temperature: float = 0.3
) -> Tuple[str, Optional[dict], Optional[str]]:
    prompt = build_smart_prompt_main(main_question, sub_questions_and_code)
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        cot_output = response.choices[0].message.content.strip()
        time.sleep(5)

        code_match = re.search(r"<code>\s*(.*?)\s*</code>", cot_output, re.DOTALL)
        code_data = None
        if code_match:
            code_str = code_match.group(1)
            try:
                code_data = json.loads(code_str)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing failed: {e}")
                print("Original JSON string:", code_str)

        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", cot_output, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else None

        return cot_output, code_data, answer

    except Exception as e:
        print(f"❌ Request failed: {e}")
        return "", None, None