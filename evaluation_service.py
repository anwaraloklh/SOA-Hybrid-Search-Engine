# evaluation_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import urllib.request
import re
import io
import os
import time
import zipfile
import numpy as np
import pandas as pd
import requests
import traceback

app = FastAPI(title="Evaluation Service")

# روابط الخدمات الخلفية للتواصل بين الخدمات (SOA)
PREPROCESSING_URL = "http://127.0.0.1:8001"
RETRIEVAL_URL = "http://127.0.0.1:8002"

# مسارات الملفات المحلية للـ Cache (لتفادي إعادة التحميل في كل مرة)
LOCAL_QRELS_PATH = "qrels_argsme.qrels"
LOCAL_TOPICS_PATH = "topics_argsme.zip"

class EvalRequest(BaseModel):
    dataset_name: str
    hybrid_type: str
    alpha: float
    k1: float
    b: float
    k: int = 10
    use_query_expansion: bool = False  # بارامتر تقييم الميزات الإضافية (قبل وبعد التوسيع)

def clean_doc_id(val):
    try:
        float_val = float(val)
        if float_val.is_integer():
            return str(int(float_val))
        return str(val).strip()
    except:
        return str(val).strip()

# =====================================================================
# دالة التحميل الذكية مع Retry ودعم Cache المحلي
# =====================================================================
def download_with_retry(url, local_path=None, max_retries=3, timeout=90):
    """
    تحاول قراءة الملف محلياً أولاً (Cache)، وإذا لم يوجد تحمّله من الإنترنت.
    عند الفشل تعيد المحاولة تلقائياً حتى max_retries مرات.
    """
    # أولاً: تحقق من وجود الملف محلياً (Cache)
    if local_path and os.path.exists(local_path):
        print(f"✅ قراءة من الكاش المحلي: {local_path}")
        with open(local_path, "rb") as f:
            return f.read()

    # ثانياً: التحميل من الإنترنت مع إعادة المحاولة
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🌐 محاولة التحميل {attempt}/{max_retries} من: {url}")
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (IR-Project/1.0)',
                    'Accept': '*/*'
                }
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()

            # حفظ الملف محلياً للمرات القادمة (Cache)
            if local_path:
                with open(local_path, "wb") as f:
                    f.write(data)
                print(f"💾 تم حفظ الملف محلياً للاستخدام اللاحق: {local_path}")

            return data

        except Exception as e:
            last_error = e
            print(f"⚠️ فشلت المحاولة {attempt}: {e}")
            if attempt < max_retries:
                wait_time = attempt * 3  # انتظار تصاعدي: 3, 6, 9 ثوان
                print(f"⏳ انتظار {wait_time} ثوان قبل إعادة المحاولة...")
                time.sleep(wait_time)

    raise Exception(f"❌ فشل التحميل بعد {max_retries} محاولات. آخر خطأ: {last_error}")

# =====================================================================
# دالات حساب مقاييس التقييم الأربعة
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
# Endpoint التقييم الرئيسي
# =====================================================================
@app.post("/evaluate")
def api_evaluate(req: EvalRequest):
    try:
        if req.dataset_name == "argsme":
            qrels_url  = "https://zenodo.org/record/6862281/files/touche2020-task1-relevance-args-me-corpus-version-1.qrels"
            topics_url = "https://zenodo.org/record/6798216/files/topics-task-1-2020.zip"
            suffix = "_2"
        else:
            raise HTTPException(
                status_code=400,
                detail="Qrels and Topics for MS MARCO are currently evaluated locally."
            )

        print(f"🔄 جاري تحضير ملفات التقييم لـ {req.dataset_name}...")

        # قراءة أول 20,000 صف لمعرفة المستندات المتوفرة
        df_temp = pd.read_csv(
            f"cleaned_dataset{suffix}.csv.gz",
            compression="gzip",
            nrows=20000
        )
        subset_docs_set = set(df_temp['doc_id'].apply(clean_doc_id))
        print(f"📦 عدد المستندات المتوفرة في العينة: {len(subset_docs_set)}")

        # ── تحميل/قراءة أحكام الصلة (مع Cache و Retry) ──────────────
        qrels_raw  = download_with_retry(qrels_url,  local_path=LOCAL_QRELS_PATH)
        qrels_text = qrels_raw.decode('utf-8')

        # ── تحميل/قراءة الاستعلامات (مع Cache و Retry) ───────────────
        zip_data   = download_with_retry(topics_url, local_path=LOCAL_TOPICS_PATH)

        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            xml_file_name = [name for name in z.namelist() if name.endswith('.xml')][0]
            with z.open(xml_file_name) as f:
                topics_xml = f.read().decode('utf-8')

        # ── تصفية أحكام الصلة لتناسب عينتنا ─────────────────────────
        qrels_dict = {}
        for line in qrels_text.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                qid          = parts[0]
                doc_id_clean = clean_doc_id(parts[2])
                relevance    = int(float(parts[3]))
                if relevance > 0 and doc_id_clean in subset_docs_set:
                    if qid not in qrels_dict:
                        qrels_dict[qid] = []
                    qrels_dict[qid].append(doc_id_clean)

        print(f"📋 عدد الاستعلامات التي لها وثائق ذات صلة في العينة: {len(qrels_dict)}")

        # ── تصفية الاستعلامات ─────────────────────────────────────────
        queries_dict = {}
        topic_blocks = re.findall(r'<topic>.*?</topic>', topics_xml, re.DOTALL)
        for block in topic_blocks:
            qid_match   = re.search(r'<number>\s*(.*?)\s*</number>', block)
            title_match = re.search(r'<title>\s*(.*?)\s*</title>',  block)
            if qid_match and title_match:
                qid   = qid_match.group(1).strip()
                title = title_match.group(1).strip()
                if qid in qrels_dict:
                    queries_dict[qid] = title
                    if len(queries_dict) >= 10:   # 10 استعلامات تكفي للتقييم
                        break

        qrels_dict = {
            qid: doc_ids
            for qid, doc_ids in qrels_dict.items()
            if qid in queries_dict
        }

        if not queries_dict:
            raise HTTPException(
                status_code=404,
                detail="لم يُعثر على استعلامات مناسبة في عينة الـ 20,000 وثيقة."
            )

        print(f"🔎 سيتم التقييم على {len(queries_dict)} استعلام.")

        # ── حساب المقاييس بالتواصل مع خدمة البحث (Port 8002) ─────────
        precisions, recalls, aps, ndcgs = [], [], [], []

        for qid, query_text in queries_dict.items():
            ground_truth    = qrels_dict[qid]
            final_query_text = query_text

            # تفعيل التوسيع عند الطلب (قبل/بعد الميزات الإضافية)
            if req.use_query_expansion:
                try:
                    res_exp      = requests.post(
                        f"{PREPROCESSING_URL}/expand",
                        json={"text": query_text},
                        timeout=10
                    ).json()
                    expanded_text = res_exp.get("expanded_text", query_text)

                    res_prep     = requests.post(
                        f"{PREPROCESSING_URL}/preprocess",
                        json={"text": expanded_text},
                        timeout=10
                    ).json()
                    final_query_text = res_prep.get("cleaned_text", query_text)
                except Exception as exp_err:
                    print(f"⚠️ فشل التوسيع للاستعلام {qid}: {exp_err} — سيُستخدم الاستعلام الأصلي.")
                    final_query_text = query_text

            # إرسال طلب البحث لخدمة الاسترجاع
            search_payload = {
                "query":          query_text,
                "cleaned_query":  final_query_text,
                "k":              req.k,
                "alpha":          req.alpha,
                "k1":             req.k1,
                "b":              req.b,
                "dataset_name":   req.dataset_name,
                "hybrid_type":    req.hybrid_type,
                "cluster_filter": "All"
            }

            try:
                res_search   = requests.post(
                    f"{RETRIEVAL_URL}/search",
                    json=search_payload,
                    timeout=30
                ).json()
                docs         = res_search.get("docs", [])
                retrieved_ids = [doc["cleaned_doc_id"] for doc in docs]
            except Exception as search_err:
                print(f"⚠️ فشل البحث للاستعلام {qid}: {search_err}")
                retrieved_ids = []

            # حساب وتجميع المقاييس
            precisions.append(precision_at_k(retrieved_ids, ground_truth, req.k))
            recalls.append(   recall_at_k(   retrieved_ids, ground_truth, req.k))
            aps.append(        average_precision(retrieved_ids, ground_truth, req.k))
            ndcgs.append(      ndcg_at_k(    retrieved_ids, ground_truth, req.k))

            print(
                f"   ✔ [{qid}] P@10={precisions[-1]:.4f} | "
                f"R@10={recalls[-1]:.4f} | "
                f"AP={aps[-1]:.4f} | "
                f"nDCG={ndcgs[-1]:.4f}"
            )

        # ── إرجاع النتائج المتوسطة ────────────────────────────────────
        result = {
            "dataset_name":    req.dataset_name,
            "hybrid_type":     req.hybrid_type,
            "num_queries":     len(queries_dict),
            "Precision_at_10": float(np.mean(precisions)),
            "Recall_at_10":    float(np.mean(recalls)),
            "MAP_at_10":       float(np.mean(aps)),
            "nDCG_at_10":      float(np.mean(ndcgs))
        }
        print(f"\n🏆 النتائج النهائية: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        print("❌ حدث خطأ في خدمة التقييم:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── endpoint مساعد لحذف الكاش يدوياً عند الحاجة ──────────────────────
@app.delete("/clear-cache")
def clear_cache():
    """يمسح الملفات المحفوظة محلياً لإجبار إعادة التحميل من Zenodo."""
    deleted = []
    for path in [LOCAL_QRELS_PATH, LOCAL_TOPICS_PATH]:
        if os.path.exists(path):
            os.remove(path)
            deleted.append(path)
    return {"deleted_files": deleted, "message": "سيتم إعادة التحميل من Zenodo في الطلب القادم."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)