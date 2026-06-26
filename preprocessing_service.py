# preprocessing_service.py
"""
خدمة المعالجة المسبقة — Preprocessing & Query Refinement Service
SOA Port: 8001
المسؤولية: تنظيف النصوص + Stemming + Lemmatization + توسيع الاستعلام
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ── تنزيل حزم NLTK المطلوبة فقط (بدون averaged_perceptron_tagger) ──
for pkg, lookup in [
    ('stopwords', 'corpora/stopwords'),
    ('punkt',     'tokenizers/punkt'),
    ('wordnet',   'corpora/wordnet'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
]:
    try:
        nltk.data.find(lookup)
    except LookupError:
        nltk.download(pkg, quiet=True)

app = FastAPI(title="Preprocessing & Refinement Service — Port 8001")

class TextQuery(BaseModel):
    text: str

# ── أدوات المعالجة ─────────────────────────────────────────────────
stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ══════════════════════════════════════════════════════════════════
# دالة التنظيف الرئيسية — Stemming + Lemmatization
# ══════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """
    خط أنابيب المعالجة المسبقة الكامل:
    1. Lowercasing
    2. إزالة الأرقام والرموز (Regex)
    3. Tokenization (NLTK)
    4. إزالة كلمات التوقف (Stop-words)
    5. Lemmatization (WordNet) — تحويل الكلمة لشكلها الأساسي
    6. Stemming (Porter)      — تحويل الكلمة لجذرها اللغوي
    """
    # 1. Lowercasing
    text = str(text).lower()

    # 2. إزالة الرموز والأرقام
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenization
    words = word_tokenize(text)

    # 4. إزالة كلمات التوقف
    words = [w for w in words if w not in stop_words and len(w) > 1]

    # 5. Lemmatization بدون POS tags (لتجنب مشكلة averaged_perceptron_tagger)
    lemmatized = [lemmatizer.lemmatize(w) for w in words]

    # 6. Stemming على الكلمات المُعالَجة بـ Lemmatization
    final_words = [stemmer.stem(w) for w in lemmatized]

    return " ".join(final_words)

# ══════════════════════════════════════════════════════════════════
# دالة توسيع الاستعلام — Query Expansion بـ WordNet
# ══════════════════════════════════════════════════════════════════
def expand_query(query: str) -> str:
    """
    توسيع الاستعلام بإضافة مرادفات من WordNet:
    - يتجاهل الكلمات القصيرة (< 4 أحرف)
    - يضيف أفضل مرادفَين لكل كلمة
    - يتوقف بعد توسيع كلمتَين كحد أقصى
    """
    words = query.strip().split()
    expanded_words = list(words)
    expanded_count = 0

    for word in words:
        if len(word) < 4:
            continue

        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_name = lemma.name().replace('_', ' ').lower()
                if (syn_name != word.lower() and
                        syn_name not in expanded_words and
                        len(syn_name.split()) == 1):
                    synonyms.append(syn_name)

        if synonyms:
            unique_syns = list(dict.fromkeys(synonyms))[:2]
            expanded_words.extend(unique_syns)
            expanded_count += 1
            if expanded_count >= 2:
                break

    return " ".join(expanded_words)

# ══════════════════════════════════════════════════════════════════
# REST API Endpoints
# ══════════════════════════════════════════════════════════════════

@app.post("/preprocess")
def api_preprocess(data: TextQuery):
    """
    تنظيف النص باستخدام:
    Lowercasing → Regex Cleaning → Tokenization →
    Stop-word Removal → Lemmatization → Stemming
    """
    if not data.text.strip():
        raise HTTPException(400, "النص فارغ")
    cleaned = clean_text(data.text)
    return {
        "original_text": data.text,
        "cleaned_text":  cleaned,
        "steps_applied": [
            "Lowercasing",
            "Regex Cleaning (remove non-alpha)",
            "NLTK Tokenization",
            "Stop-word Removal",
            "WordNet Lemmatization",
            "Porter Stemming",
        ]
    }

@app.post("/expand")
def api_expand(data: TextQuery):
    """
    توسيع الاستعلام بمرادفات WordNet.
    الاستخدام: وضع Advanced Search في الواجهة.
    """
    if not data.text.strip():
        raise HTTPException(400, "الاستعلام فارغ")
    expanded = expand_query(data.text)
    return {
        "original_query": data.text,
        "expanded_text":  expanded,
        "added_terms":    [w for w in expanded.split()
                           if w not in data.text.split()],
    }

@app.get("/health")
def health_check():
    """فحص حالة الخدمة."""
    return {
        "service":    "Preprocessing Service",
        "port":       8001,
        "status":     "running",
        "components": {
            "stemmer":         "PorterStemmer (NLTK)",
            "lemmatizer":      "WordNetLemmatizer (NLTK)",
            "stop_words":      len(stop_words),
            "query_expansion": "WordNet Synonyms",
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Preprocessing Service — Port 8001")
    print("📋 Endpoints:")
    print("   POST /preprocess  — تنظيف النص (Lemmatization + Stemming)")
    print("   POST /expand      — توسيع الاستعلام بالمرادفات")
    print("   GET  /health      — حالة الخدمة")
    uvicorn.run(app, host="127.0.0.1", port=8001)