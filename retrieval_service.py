# retrieval_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import torch
import traceback
from sentence_transformers import util

# --- تم نقل الاستيراد هنا في أعلى الملف لحل خطأ الـ Pickle والمساحة الاسمية للأبد ---
from sklearn.feature_extraction.text import TfidfVectorizer

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
    cluster_filter: str

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
current_dataset_name = None
df_subset = None
bm25_model = None
model = None
embeddings = None
tfidf_vectorizer = None
tfidf_matrix = None
subset_size = 20000

def load_resources(dataset_name):
    global current_dataset_name, df_subset, bm25_model, model, embeddings, tfidf_vectorizer, tfidf_matrix
    
    if current_dataset_name == dataset_name and df_subset is not None and 'cluster_id' in df_subset.columns:
        return # إذا كانت المجموعة محملة وبها كلوستر فلا نعيد تحميلها
        
    print(f"🔄 جاري تحميل نماذج وفهارس مجموعة البيانات: ({dataset_name})...")
    suffix = "_2" if dataset_name == "argsme" else ""
    
    df_temp = pd.read_csv(f"cleaned_dataset{suffix}.csv.gz", compression="gzip", nrows=subset_size)
    df_temp['cleaned_text'] = df_temp['cleaned_text'].fillna('')
    df_subset = df_temp.copy()
    
    # تحميل فهارس TF-IDF (VSM) بنجاح بعد تأمين الاستيراد العالمي
    with open(f'tfidf_model{suffix}.pkl', 'rb') as f:
        tfidf_vectorizer, raw_tfidf_matrix = pickle.load(f)
    # قص مصفوفة TF-IDF لتطابق أول 20,000 مستند
    tfidf_matrix = raw_tfidf_matrix[:subset_size]
    
    with open(f'bm25_model{suffix}.pkl', 'rb') as f:
        bm25_model = pickle.load(f)
        
    with open(f'bert_embeddings{suffix}.pkl', 'rb') as f:
        bert_data = pickle.load(f)
        model = bert_data['model']
        raw_embeddings = bert_data['embeddings']
        
    # تأمين تحويل المتجهات الدلالية
    if isinstance(raw_embeddings, np.ndarray):
        embeddings = torch.from_numpy(raw_embeddings).cpu()
    else:
        embeddings = raw_embeddings.cpu()
        
    current_dataset_name = dataset_name
    print(f"✅ تم تحميل مصادر مجموعة ({dataset_name}) بنجاح!")

@app.post("/search")
def api_search(req: SearchRequest):
    try:
        load_resources(req.dataset_name)
        
        # تطبيق قيم k1 و b التفاعلية لـ BM25
        bm25_model.k1 = float(req.k1)
        bm25_model.b = float(req.b)
        subset_size_local = len(df_subset)
        
        # تصفية المستندات والمتجهات بناءً على الكلوستر المحدد
        if req.cluster_filter != "All" and 'cluster_id' in df_subset.columns:
            cluster_map = {"Topic 1": 0, "Topic 2": 1, "Topic 3": 2, "Topic 4": 3, "Topic 5": 4}
            target_cluster = cluster_map.get(req.cluster_filter, -1)
            matching_indices = np.where(df_subset['cluster_id'] == target_cluster)[0]
        else:
            matching_indices = np.arange(subset_size_local)
            
        if len(matching_indices) == 0:
            return {"docs": [], "scores": []}
            
        # =====================================================================
        # أ. نمط البحث باستخدام TF-IDF (VSM) الصرف
        # =====================================================================
        if req.hybrid_type == "TF-IDF":
            # تمثيل الاستعلام كمتجه جيب تمام
            query_vec = tfidf_vectorizer.transform([req.cleaned_query])
            sub_matrix = tfidf_matrix[matching_indices]
            scores_array = np.array((sub_matrix * query_vec.T).toarray()).flatten()
            
            top_filtered_indices = np.argsort(scores_array)[-req.k:][::-1].copy()
            final_mapped_indices = matching_indices[top_filtered_indices]
            scores = scores_array[top_filtered_indices].tolist()
            
        # =====================================================================
        # ب. نمط البحث باستخدام BM25 الصرف
        # =====================================================================
        elif req.hybrid_type == "BM25":
            bm25_scores = np.array(bm25_model.get_scores(req.cleaned_query.split()))[matching_indices]
            top_filtered_indices = np.argsort(bm25_scores)[-req.k:][::-1].copy()
            final_mapped_indices = matching_indices[top_filtered_indices]
            scores = bm25_scores[top_filtered_indices].tolist()
            
        # =====================================================================
        # ج. نمط البحث باستخدام BERT الصرف (الدلالي)
        # =====================================================================
        elif req.hybrid_type == "BERT":
            candidate_embeddings = embeddings[matching_indices]
            query_embedding = model.encode(req.cleaned_query, convert_to_tensor=True).cpu()
            bert_scores = util.cos_sim(query_embedding, candidate_embeddings)[0].numpy()
            
            top_filtered_indices = np.argsort(bert_scores)[-req.k:][::-1].copy()
            final_mapped_indices = matching_indices[top_filtered_indices]
            scores = bert_scores[top_filtered_indices].tolist()
            
        # =====================================================================
        # د. نمط البحث الهجين المتوازي
        # =====================================================================
        elif req.hybrid_type == "Parallel":
            bm25_scores = np.array(bm25_model.get_scores(req.cleaned_query.split()))[matching_indices]
            candidate_embeddings = embeddings[matching_indices]
            query_embedding = model.encode(req.cleaned_query, convert_to_tensor=True).cpu()
            bert_scores = util.cos_sim(query_embedding, candidate_embeddings)[0].numpy()
            
            bm25_norm = normalize_scores(bm25_scores)
            bert_norm = normalize_scores(bert_scores)
            final_scores = (req.alpha * bert_norm) + ((1 - req.alpha) * bm25_norm)
            
            top_filtered_indices = np.argsort(final_scores)[-req.k:][::-1].copy()
            final_mapped_indices = matching_indices[top_filtered_indices]
            scores = final_scores[top_filtered_indices].tolist()
            
        # =====================================================================
        # هـ. نمط البحث الهجين التسلسلي
        # =====================================================================
        else:
            bm25_scores = np.array(bm25_model.get_scores(req.cleaned_query.split()))[matching_indices]
            candidate_pool = min(100, len(matching_indices))
            top_filtered_candidate_indices = np.argsort(bm25_scores)[-candidate_pool:][::-1].copy()
            
            actual_indices = matching_indices[top_filtered_candidate_indices]
            actual_indices_list = actual_indices.tolist()
            
            candidate_embeddings = embeddings[actual_indices_list]
            query_embedding = model.encode(req.cleaned_query, convert_to_tensor=True).cpu()
            bert_candidate_scores = util.cos_sim(query_embedding, candidate_embeddings)[0].numpy()
            
            candidate_bm25_scores = bm25_scores[top_filtered_candidate_indices]
            bm25_norm = normalize_scores(candidate_bm25_scores)
            bert_norm = normalize_scores(bert_candidate_scores)
            
            final_candidate_scores = (req.alpha * bert_norm) + ((1 - req.alpha) * bm25_norm)
            final_top_indices = np.argsort(final_candidate_scores)[-req.k:][::-1].copy()
            final_mapped_indices = actual_indices[final_top_indices]
            scores = final_candidate_scores[final_top_indices].tolist()
            
        results = df_subset.iloc[final_mapped_indices][['doc_id', 'text', 'cluster_id']].copy()
        results['cleaned_doc_id'] = results['doc_id'].apply(clean_doc_id)
        
        return {
            "docs": results.to_dict(orient="records"),
            "scores": scores
        }
    except Exception as e:
        print("❌ حدث خطأ داخلي في خادم البحث:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)