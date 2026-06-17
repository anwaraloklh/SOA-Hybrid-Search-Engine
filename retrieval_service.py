# retrieval_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import util

app = FastAPI(title="Retrieval & Ranking Service")

class SearchRequest(BaseModel):
    query: str
    cleaned_query: str
    k: int
    alpha: float
    k1: float
    b: float
    dataset_name: str
    hybrid_type: str

# دالة تسوية القيم
def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

# دالة تنظيف معرفات المستندات
def clean_doc_id(val):
    try:
        float_val = float(val)
        if float_val.is_integer():
            return str(int(float_val))
        return str(val).strip()
    except:
        return str(val).strip()

# تحميل النماذج بشكل آلي وديناميكي
resources = {}
def load_resources(dataset_name):
    if dataset_name in resources:
        return resources[dataset_name]
    
    suffix = "_2" if dataset_name == "argsme" else ""
    df_temp = pd.read_csv(f"cleaned_dataset{suffix}.csv.gz", compression="gzip", nrows=20000)
    df_temp['cleaned_text'] = df_temp['cleaned_text'].fillna('')
    df_subset = df_temp.copy()
    
    with open(f'bm25_model{suffix}.pkl', 'rb') as f:
        bm25_model = pickle.load(f)
        
    with open(f'bert_embeddings{suffix}.pkl', 'rb') as f:
        bert_data = pickle.load(f)
        model = bert_data['model']
        embeddings = bert_data['embeddings']
        
    resources[dataset_name] = (df_subset, bm25_model, model, embeddings)
    return resources[dataset_name]

@app.post("/search")
def api_search(req: SearchRequest):
    try:
        df_subset, bm25_model, model, embeddings = load_resources(req.dataset_name)
        
        # تطبيق قيم k1 و b التفاعلية
        bm25_model.k1 = req.k1
        bm25_model.b = req.b
        subset_size = 20000
        
        if "Parallel" in req.hybrid_type:
            # البحث التفرعي
            bm25_scores = np.array(bm25_model.get_scores(req.cleaned_query.split()))[:subset_size]
            query_embedding = model.encode(req.cleaned_query, convert_to_tensor=True)
            bert_scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
            
            bm25_norm = normalize_scores(bm25_scores)
            bert_norm = normalize_scores(bert_scores)
            final_scores = (req.alpha * bert_norm) + ((1 - req.alpha) * bm25_norm)
            top_indices = np.argsort(final_scores)[-req.k:][::-1].copy()
            
            results = df_subset.iloc[top_indices][['doc_id', 'text']].copy()
            results['cleaned_doc_id'] = results['doc_id'].apply(clean_doc_id)
            scores = final_scores[top_indices].tolist()
            
        else:
            # البحث التسلسلي
            bm25_scores = np.array(bm25_model.get_scores(req.cleaned_query.split()))[:subset_size]
            top_candidate_indices = np.argsort(bm25_scores)[-100:][::-1].copy()
            
            candidate_embeddings = embeddings[top_candidate_indices]
            query_embedding = model.encode(req.cleaned_query, convert_to_tensor=True)
            bert_candidate_scores = util.cos_sim(query_embedding, candidate_embeddings)[0].cpu().numpy()
            
            candidate_bm25_scores = bm25_scores[top_candidate_indices]
            bm25_norm = normalize_scores(candidate_bm25_scores)
            bert_norm = normalize_scores(bert_candidate_scores)
            
            final_candidate_scores = (req.alpha * bert_norm) + ((1 - req.alpha) * bm25_norm)
            final_top_indices = np.argsort(final_candidate_scores)[-req.k:][::-1].copy()
            final_mapped_indices = top_candidate_indices[final_top_indices]
            
            results = df_subset.iloc[final_mapped_indices][['doc_id', 'text']].copy()
            results['cleaned_doc_id'] = results['doc_id'].apply(clean_doc_id)
            scores = final_candidate_scores[final_top_indices].tolist()
            
        return {
            "docs": results.to_dict(orient="records"),
            "scores": scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)