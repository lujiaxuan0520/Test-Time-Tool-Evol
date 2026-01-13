import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from tool_library import ToolLibrary

def select_tool(
    query: str,
    model_dir: str,
    reranker_dir: str,
    tool_library: ToolLibrary,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    tool_names, tool_descriptions, tool_embs = tool_library.get_embeddings()
    if tool_embs is None or len(tool_names) == 0:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_dir, trust_remote_code=True)
    embedder.to(device)
    query_emb = embedder.encode([query], normalize_embeddings=True, device=device)
    embedder.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    query_emb = torch.tensor(query_emb).to('cpu')
    tool_embs_tensor = torch.tensor(tool_embs).to('cpu')
    similarities = torch.nn.functional.cosine_similarity(query_emb, tool_embs_tensor)
    k = min(top_k, len(similarities))
    top_k_indices = torch.topk(similarities, k, largest=True).indices.tolist()

    rerank_pairs = [(query, tool_descriptions[i]) for i in top_k_indices]
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_dir)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_dir)
    reranker_model.to(device)
    reranker_model.eval()

    with torch.no_grad():
        inputs = reranker_tokenizer(
            rerank_pairs,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        ).to(device)
        scores = reranker_model(**inputs).logits.view(-1).sigmoid().cpu().numpy()

    reranked_results = [
        (tool_names[idx], float(score))
        for idx, score in zip(top_k_indices, scores)
    ]
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    return reranked_results