# preprocessing_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer

# تنزيل حزم NLTK
for package in ['stopwords', 'punkt', 'wordnet']:
    try:
        nltk.data.find(f'corpora/{package}' if package != 'punkt' else f'tokenizers/{package}')
    except LookupError:
        nltk.download(package)

app = FastAPI(title="Preprocessing & Refinement Service")

class TextQuery(BaseModel):
    text: str

# دالة المعالجة الأساسية
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    return " ".join([stemmer.stem(w) for w in words])

# دالة توسيع الاستعلام
def expand_query(query):
    words = query.strip().split()
    expanded_words = list(words)
    count = 0
    for word in words:
        if len(word) < 4:
            continue
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                syn_name = lemma.name().replace('_', ' ').lower()
                if syn_name != word.lower() and syn_name not in expanded_words:
                    synonyms.append(syn_name)
        if synonyms:
            expanded_words.extend(synonyms[:2])
            count += 1
            if count >= 2:
                break
    return " ".join(expanded_words)

@app.post("/preprocess")
def api_preprocess(data: TextQuery):
    return {"cleaned_text": clean_text(data.text)}

@app.post("/expand")
def api_expand(data: TextQuery):
    return {"expanded_text": expand_query(data.text)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
    