# evaluation_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.request
import re
import io
import zipfile
import numpy as np
import pandas as pd
import requests
import traceback

app = FastAPI(title="Evaluation Service")

# رابط خدمة البحث الخلفية للتواصل بين الخدمات (Port 8002)
RETRIEVAL_URL = "http://127.0.0.1:8002"

class EvalRequest(BaseModel):
    dataset_name: str
    hybrid_type: str
    alpha: float
    k1: float
    b: float
    k: int = 10

# دالة تنظيف معرفات المستندات لضمان معالجة الأرقام العشرية (مثل 2.0)
def clean_doc_id(val):
    try:
        float_val = float(val)
        if float_val.is_integer():
            return str(int(float_val))
        return str(val).strip()
    except:
        return str(val).strip()

# =====================================================================
# دالات حساب مقاييس التقييم الأكاديمية الأربعة
# =====================================================================
def precision_at_k(retrieved_ids, ground_truth_ids, k=10):
    retrieved_k = retrieved_ids[:k]
    if len(retrieved_k) == 0:
        return 0.0
    relevant_retrieved = sum([1 for doc_id in retrieved_k if clean_doc_id(doc_id) in ground_truth_ids])
    return relevant_retrieved / k

def recall_at_k(retrieved_ids, ground_truth_ids, k=10):
    retrieved_k = retrieved_ids[:k]
    actual_relevant = len(ground_truth_ids)
    if actual_relevant == 0:
        return 0.0
    relevant_retrieved = sum([1 for doc_id in retrieved_k if clean_doc_id(doc_id) in ground_truth_ids])
    return relevant_retrieved / actual_relevant

def average_precision(retrieved_ids, ground_truth_ids, k=10):
    retrieved_k = retrieved_ids[:k]
    actual_relevant = len(ground_truth_ids)
    if actual_relevant == 0:
        return 0.0
    
    ap_sum = 0.0
    relevant_count = 0
    for rank, doc_id in enumerate(retrieved_k):
        if clean_doc_id(doc_id) in ground_truth_ids:
            relevant_count += 1
            precision = relevant_count / (rank + 1)
            ap_sum += precision
            
    return ap_sum / min(k, actual_relevant) if actual_relevant > 0 else 0.0

def ndcg_at_k(retrieved_ids, ground_truth_ids, k=10):
    retrieved_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_k):
        if clean_doc_id(doc_id) in ground_truth_ids:
            dcg += 1.0 / np.log2(rank + 2)
            
    actual_relevant = len(ground_truth_ids)
    idcg = 0.0
    for rank in range(min(k, actual_relevant)):
        idcg += 1.0 / np.log2(rank + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

# =====================================================================
# REST API Endpoint لتشغيل التقييم من الويب أو بوابة المستخدم
# =====================================================================
@app.post("/evaluate")
def api_evaluate(req: EvalRequest):
    try:
        # 1. تحديد روابط التحميل بناءً على المجموعة المطلوبة
        if req.dataset_name == "argsme":
            qrels_url = "https://zenodo.org/record/6862281/files/touche2020-task1-relevance-args-me-corpus-version-1.qrels"
            topics_url = "https://zenodo.org/record/6798216/files/topics-task-1-2020.zip"
            suffix = "_2"
        else:
            # هنا يمكنك وضع روابط MS MARCO بنفس الطريقة إذا أردت تقييمها
            raise HTTPException(status_code=400, detail="Qrels and Topics for MS MARCO are currently evaluated locally.")

        print(f"🔄 جاري قراءة المستندات المصفاة وتنزيل ملفات التقييم لـ {req.dataset_name} من Zenodo...")
        
        # قراءة أول 20,000 صف لمعرفة المستندات المتوفرة للتصفية
        df_temp = pd.read_csv(f"cleaned_dataset{suffix}.csv.gz", compression="gzip", nrows=20000)
        subset_docs_set = set(df_temp['doc_id'].apply(clean_doc_id))

        # تنزيل أحكام الصلة
        with urllib.request.urlopen(qrels_url) as response:
            qrels_text = response.read().decode('utf-8')
            
        # تنزيل الاستعلامات
        with urllib.request.urlopen(topics_url) as response:
            zip_data = response.read()
            
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            xml_file_name = [name for name in z.namelist() if name.endswith('.xml')][0]
            with z.open(xml_file_name) as f:
                topics_xml = f.read().decode('utf-8')

        # تصفية أحكام الصلة لتناسب عينتك
        qrels_dict = {}
        for line in qrels_text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                qid = parts[0]
                doc_id_clean = clean_doc_id(parts[2])
                relevance = int(float(parts[3]))
                if relevance > 0 and doc_id_clean in subset_docs_set:
                    if qid not in qrels_dict:
                        qrels_dict[qid] = []
                    qrels_dict[qid].append(doc_id_clean)

        # تصفية الاستعلامات
        queries_dict = {}
        topic_blocks = re.findall(r'<topic>.*?</topic>', topics_xml, re.DOTALL)
        for block in topic_blocks:
            qid_match = re.search(r'<number>\s*(.*?)\s*</number>', block)
            title_match = re.search(r'<title>\s*(.*?)\s*</title>', block)
            if qid_match and title_match:
                qid = qid_match.group(1).strip()
                title = title_match.group(1).strip()
                if qid in qrels_dict:
                    queries_dict[qid] = title
                    if len(queries_dict) >= 10: # نكتفي بـ 10 استعلامات للتقييم
                        break
                        
        qrels_dict = {qid: doc_ids for qid, doc_ids in qrels_dict.items() if qid in queries_dict}

        # 2. حساب المقاييس الأربعة بالتواصل مع خدمة البحث (Port 8002) عبر الـ REST API!
        precisions = []
        recalls = []
        aps = []
        ndcgs = []

        print(f"🔄 جاري إرسال طلبات البحث لخدمة البحث (Port 8002) لحساب المقاييس...")
        for qid, query_text in queries_dict.items():
            ground_truth = qrels_dict[qid]
            
            # إرسال طلب HTTP POST للبحث عن الاستعلام واستلام المستندات المسترجعة
            search_payload = {
                "query": query_text,
                "cleaned_query": query_text, # خدمة البحث تتولى المعالجة اللغوية تلقائياً
                "k": req.k,
                "alpha": req.alpha,
                "k1": req.k1,
                "b": req.b,
                "dataset_name": req.dataset_name,
                "hybrid_type": req.hybrid_type,
                "cluster_filter": "All"
            }
            
            res_search = requests.post(f"{RETRIEVAL_URL}/search", json=search_payload).json()
            docs = res_search.get("docs", [])
            retrieved_ids = [doc["cleaned_doc_id"] for doc in docs]
            
            # حساب المقاييس
            precisions.append(precision_at_k(retrieved_ids, ground_truth, req.k))
            recalls.append(recall_at_k(retrieved_ids, ground_truth, req.k))
            aps.append(average_precision(retrieved_ids, ground_truth, req.k))
            ndcgs.append(ndcg_at_k(retrieved_ids, ground_truth, req.k))

        return {
            "dataset_name": req.dataset_name,
            "hybrid_type": req.hybrid_type,
            "Precision_at_10": float(np.mean(precisions)),
            "Recall_at_10": float(np.mean(recalls)),
            "MAP_at_10": float(np.mean(aps)),
            "nDCG_at_10": float(np.mean(ndcgs))
        }

    except Exception as e:
        print("❌ حدث خطأ في خدمة التقييم:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)