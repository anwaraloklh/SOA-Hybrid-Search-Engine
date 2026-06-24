# build_tfidf.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

print("🔄 جاري بناء وحفظ مصفوفات TF-IDF للمجموعتين محلياً داخل مجلد المشروع...")

# 1. بناء وحفظ نموذج TF-IDF للمجموعة الأولى MS MARCO
df1 = pd.read_csv("cleaned_dataset.csv.gz", compression="gzip", nrows=20000)
df1['cleaned_text'] = df1['cleaned_text'].fillna('')
vectorizer1 = TfidfVectorizer()
matrix1 = vectorizer1.fit_transform(df1['cleaned_text'])

with open('tfidf_model.pkl', 'wb') as f:
    pickle.dump((vectorizer1, matrix1), f)
print("✅ تم بناء وحفظ tfidf_model.pkl لـ MS MARCO بنجاح.")

# 2. بناء وحفظ نموذج TF-IDF للمجموعة الثانية argsme
df2 = pd.read_csv("cleaned_dataset_2.csv.gz", compression="gzip", nrows=20000)
df2['cleaned_text'] = df2['cleaned_text'].fillna('')
vectorizer2 = TfidfVectorizer()
matrix2 = vectorizer2.fit_transform(df2['cleaned_text'])

with open('tfidf_model_2.pkl', 'wb') as f:
    pickle.dump((vectorizer2, matrix2), f)
print("✅ تم بناء وحفظ tfidf_model_2.pkl لـ argsme بنجاح.")

print("\n🎉 تم بناء وتوليد ملفات TF-IDF بنجاح تام!")