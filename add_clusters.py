# add_clusters.py
import pandas as pd
import pickle
from sklearn.cluster import KMeans

print("🔄 جاري تشغيل الكلوسترينغ لتحديث الملفات المحلية داخل مجلد VS Code...")

# 1. تحديث المجموعة الأولى MS MARCO
df1 = pd.read_csv("cleaned_dataset.csv.gz", compression="gzip", nrows=20000)
with open('bert_embeddings.pkl', 'rb') as f:
    embeddings_1 = pickle.load(f)['embeddings']
if hasattr(embeddings_1, 'cpu'):
    embeddings_1 = embeddings_1.cpu().numpy()
    
kmeans_1 = KMeans(n_clusters=5, random_state=42, n_init='auto')
df1['cluster_id'] = kmeans_1.fit_predict(embeddings_1)
df1.to_csv("cleaned_dataset.csv.gz", index=False, compression="gzip", encoding="utf-8")
print("✅ تم تحديث كلوستر MS MARCO محلياً.")

# 2. تحديث المجموعة الثانية argsme
df2 = pd.read_csv("cleaned_dataset_2.csv.gz", compression="gzip", nrows=20000)
with open('bert_embeddings_2.pkl', 'rb') as f:
    embeddings_2 = pickle.load(f)['embeddings']
if hasattr(embeddings_2, 'cpu'):
    embeddings_2 = embeddings_2.cpu().numpy()
    
kmeans_2 = KMeans(n_clusters=5, random_state=42, n_init='auto')
df2['cluster_id'] = kmeans_2.fit_predict(embeddings_2)
df2.to_csv("cleaned_dataset_2.csv.gz", index=False, compression="gzip", encoding="utf-8")
print("✅ تم تحديث كلوستر argsme محلياً.")
print("\n🎉 تم تحديث جميع الملفات بنجاح تام!")