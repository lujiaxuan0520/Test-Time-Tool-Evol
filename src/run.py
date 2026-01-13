import json
from sentence_transformers import SentenceTransformer
from tool_library import ToolLibrary
from main_pipeline import main_pipeline

if __name__ == "__main__":
    tool_path = r"/root/autodl-tmp/tool/sub_tool_250.json"
    all_datas_path = r"/root/autodl-tmp/data_qa/datas_random.json"
    output_file = "/root/autodl-tmp/tool/decompose_random_datas_answer_1228.jsonl"

    cot_output_api_key = "xxxxx"
    cot_output_base_url = "xxxxx"
    fun_call_api_key = "xxxxx"
    fun_call_base_url = "xxxxx"

    python_bert_model_dir = r"/root/autodl-tmp/bert"
    bge_emb_model_dir = r"/root/autodl-tmp/bge"
    reranker_model_dir = r"/root/autodl-tmp/rerank"

    with open(all_datas_path, 'r', encoding='utf-8') as f:
        all_datas = json.load(f)

    # 初始化 embedder
    embedder = SentenceTransformer(bge_emb_model_dir, trust_remote_code=True)

    tool_library = ToolLibrary(
        max_size=250,
        save_path=tool_path,
        embedder=embedder
    )

    main_pipeline(
        all_datas=all_datas,
        tool_library=tool_library,
        fun_call_api_key=fun_call_api_key,
        fun_call_base_url=fun_call_base_url,
        cot_output_api_key=cot_output_api_key,
        cot_output_base_url=cot_output_base_url,
        model_dir=bge_emb_model_dir,
        reranker_dir=reranker_model_dir,
        model_path=python_bert_model_dir,
        output_file=output_file
    )