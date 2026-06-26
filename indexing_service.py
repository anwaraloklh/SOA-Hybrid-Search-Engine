# indexing_service.py
"""
خدمة الفهرسة المستقلة — Indexing Service
SOA Port: 8004
المسؤولية: بناء وإدارة جميع الفهارس (TF-IDF, BM25, BERT Embeddings)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

app = FastAPI(title="Indexing Service — Port 8004")

# ── حالة الفهارس المحمّلة ──────────────────────────────────────────
index_status = {
    "argsme":  {"tfidf": False, "bm25": False, "bert": False},
    "msmarco": {"tfidf": False, "bm25": False, "bert": False},
}

class IndexRequest(BaseModel):
    dataset_name: str   # "argsme" أو "msmarco"
    index_type: str     # "tfidf" أو "bm25" أو "bert" أو "all"

class IndexStatusResponse(BaseModel):
    dataset_name: str
    tfidf_ready: bool
    bm25_ready:  bool
    bert_ready:  bool
    tfidf_size:  str
    bm25_size:   str
    bert_size:   str

# ── Helper: حجم الملف ─────────────────────────────────────────────
def file_size(path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size > 1024*1024:
            return f"{size/1024/1024:.1f} MB"
        return f"{size/1024:.0f} KB"
    return "غير موجود"

def get_suffix(dataset_name):
    return "_2" if dataset_name == "argsme" else ""

# ══════════════════════════════════════════════════════════════════
# GET /status — حالة الفهارس
# ══════════════════════════════════════════════════════════════════
@app.get("/status/{dataset_name}", response_model=IndexStatusResponse)
def get_index_status(dataset_name: str):
    """
    يُرجع حالة كل فهرس للداتاسيت المحدد — هل موجود ومحدّث.
    """
    if dataset_name not in ["argsme", "msmarco"]:
        raise HTTPException(400, detail="dataset_name يجب أن يكون argsme أو msmarco")

    suffix = get_suffix(dataset_name)

    tfidf_path = f"tfidf_model{suffix}.pkl"
    bm25_path  = f"bm25_model{suffix}.pkl"
    bert_path  = f"bert_embeddings{suffix}.pkl"

    return IndexStatusResponse(
        dataset_name=dataset_name,
        tfidf_ready=os.path.exists(tfidf_path),
        bm25_ready= os.path.exists(bm25_path),
        bert_ready= os.path.exists(bert_path),
        tfidf_size= file_size(tfidf_path),
        bm25_size=  file_size(bm25_path),
        bert_size=  file_size(bert_path),
    )

# ══════════════════════════════════════════════════════════════════
# POST /build — بناء الفهرس (TF-IDF أو BM25)
# ══════════════════════════════════════════════════════════════════
@app.post("/build")
def build_index(req: IndexRequest):
    """
    يبني الفهرس المطلوب للداتاسيت المحدد ويحفظه في ملف pkl.
    BERT غير متاح هنا لأنه يحتاج GPU وحُسب مسبقاً.

    أنواع الفهارس المدعومة:
    - tfidf: TF-IDF Inverted Index (sparse matrix)
    - bm25:  BM25 Probabilistic Index
    - all:   الاثنان معاً
    """
    if req.dataset_name not in ["argsme", "msmarco"]:
        raise HTTPException(400, "dataset_name غير صحيح")

    if req.index_type not in ["tfidf", "bm25", "all"]:
        raise HTTPException(400, "index_type يجب أن يكون tfidf أو bm25 أو all")

    suffix = get_suffix(req.dataset_name)
    csv_path = f"cleaned_dataset{suffix}.csv.gz"

    if not os.path.exists(csv_path):
        raise HTTPException(404, detail=f"ملف الداتاسيت غير موجود: {csv_path}")

    try:
        print(f"📂 تحميل الداتاسيت: {csv_path} (أول 20,000 صف)...")
        df = pd.read_csv(csv_path, compression="gzip", nrows=20000)
        df['cleaned_text'] = df['cleaned_text'].fillna('')
        corpus = df['cleaned_text'].tolist()
        print(f"✅ تم تحميل {len(corpus)} وثيقة")

        results = {}

        # ── بناء فهرس TF-IDF ────────────────────────────────────
        if req.index_type in ["tfidf", "all"]:
            print("🔨 جاري بناء فهرس TF-IDF...")
            vectorizer = TfidfVectorizer(
                max_features=50000,   # أهم 50,000 مصطلح
                min_df=2,             # تجاهل المصطلحات النادرة جداً
                sublinear_tf=True,    # تخفيف تأثير التكرار العالي
            )
            tfidf_matrix = vectorizer.fit_transform(corpus)
            tfidf_path = f"tfidf_model{suffix}.pkl"
            with open(tfidf_path, "wb") as f:
                pickle.dump((vectorizer, tfidf_matrix), f)
            vocab_size = len(vectorizer.vocabulary_)
            results["tfidf"] = {
                "status": "✅ تم البناء بنجاح",
                "documents": len(corpus),
                "vocabulary_size": vocab_size,
                "matrix_shape": str(tfidf_matrix.shape),
                "file_size": file_size(tfidf_path),
                "index_type": "Sparse Inverted Index (TF-IDF Matrix)",
            }
            print(f"✅ TF-IDF جاهز: {vocab_size} مصطلح، {tfidf_matrix.shape}")

        # ── بناء فهرس BM25 ──────────────────────────────────────
        if req.index_type in ["bm25", "all"]:
            print("🔨 جاري بناء فهرس BM25...")
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(
                tokenized_corpus,
                k1=1.5,    # قيمة افتراضية — تُعدَّل ديناميكياً في خدمة البحث
                b=0.75,
            )
            bm25_path = f"bm25_model{suffix}.pkl"
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25, f)
            results["bm25"] = {
                "status": "✅ تم البناء بنجاح",
                "documents": len(corpus),
                "avg_doc_length": round(bm25.avgdl, 2),
                "file_size": file_size(bm25_path),
                "index_type": "Probabilistic BM25 Index (rank_bm25)",
            }
            print(f"✅ BM25 جاهز: avg_doc_length={bm25.avgdl:.2f}")

        return {
            "dataset_name": req.dataset_name,
            "index_type": req.index_type,
            "results": results,
            "message": f"✅ تمت فهرسة {len(corpus)} وثيقة بنجاح"
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

# ══════════════════════════════════════════════════════════════════
# GET /info — معلومات تقنية عن الفهرس
# ══════════════════════════════════════════════════════════════════
@app.get("/info/{dataset_name}/{index_type}")
def get_index_info(dataset_name: str, index_type: str):
    """
    يُرجع معلومات تقنية مفصّلة عن الفهرس المحفوظ.
    """
    suffix = get_suffix(dataset_name)

    if index_type == "tfidf":
        path = f"tfidf_model{suffix}.pkl"
        if not os.path.exists(path):
            raise HTTPException(404, "الفهرس غير موجود — استدعِ /build أولاً")
        with open(path, "rb") as f:
            vectorizer, matrix = pickle.load(f)
        return {
            "index_type": "TF-IDF Inverted Index",
            "description": "Sparse matrix — كل صف وثيقة، كل عمود مصطلح، القيمة وزن TF-IDF",
            "vocabulary_size": len(vectorizer.vocabulary_),
            "documents": matrix.shape[0],
            "matrix_shape": str(matrix.shape),
            "sparsity": f"{(1 - matrix.nnz / (matrix.shape[0]*matrix.shape[1]))*100:.2f}%",
            "file_size": file_size(path),
            "similarity_metric": "Cosine Similarity",
        }

    elif index_type == "bm25":
        path = f"bm25_model{suffix}.pkl"
        if not os.path.exists(path):
            raise HTTPException(404, "الفهرس غير موجود — استدعِ /build أولاً")
        with open(path, "rb") as f:
            bm25 = pickle.load(f)
        return {
            "index_type": "BM25 Probabilistic Index",
            "description": "فهرس احتمالي يُطبّع تكرار المصطلح وطول الوثيقة",
            "documents": bm25.corpus_size,
            "avg_doc_length": round(bm25.avgdl, 2),
            "k1": bm25.k1,
            "b": bm25.b,
            "file_size": file_size(path),
            "similarity_metric": "BM25 Score (not normalized)",
        }

    elif index_type == "bert":
        path = f"bert_embeddings{suffix}.pkl"
        if not os.path.exists(path):
            raise HTTPException(404, "الفهرس غير موجود")
        return {
            "index_type": "BERT Dense Vector Index",
            "description": "متجهات كثيفة 384 بُعداً لكل وثيقة — مُحسَبة مسبقاً",
            "file_size": file_size(path),
            "similarity_metric": "Cosine Similarity",
        }

    raise HTTPException(400, "index_type غير صحيح")

# ══════════════════════════════════════════════════════════════════
# DELETE /clear — حذف الفهرس لإعادة البناء
# ══════════════════════════════════════════════════════════════════
@app.delete("/clear/{dataset_name}/{index_type}")
def clear_index(dataset_name: str, index_type: str):
    """يحذف ملف الفهرس لإجبار إعادة البناء."""
    suffix = get_suffix(dataset_name)
    paths = {
        "tfidf": f"tfidf_model{suffix}.pkl",
        "bm25":  f"bm25_model{suffix}.pkl",
        "bert":  f"bert_embeddings{suffix}.pkl",
    }
    if index_type not in paths:
        raise HTTPException(400, "index_type غير صحيح")
    path = paths[index_type]
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"✅ تم حذف {path}"}
    return {"message": f"الملف غير موجود: {path}"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Indexing Service — Port 8004")
    print("📋 Endpoints:")
    print("   GET  /status/{dataset_name}         — حالة الفهارس")
    print("   POST /build                          — بناء الفهرس")
    print("   GET  /info/{dataset_name}/{type}     — معلومات تقنية")
    print("   DELETE /clear/{dataset_name}/{type}  — حذف الفهرس")
    uvicorn.run(app, host="127.0.0.1", port=8004)