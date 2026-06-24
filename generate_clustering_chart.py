# generate_clustering_chart.py
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("🔄 جاري خفض الأبعاد ورسم مخطط المجموعات الدلالية الـ 5 (Clustering Chart)...")

try:
    # 1. تحميل أول 20,000 صف من المجموعة الثانية
    df = pd.read_csv("cleaned_dataset_2.csv.gz", compression="gzip", nrows=20000)
    
    # 2. تحميل المتجهات الدلالية
    with open('bert_embeddings_2.pkl', 'rb') as f:
        embeddings = pickle.load(f)['embeddings']
        
    if hasattr(embeddings, 'cpu'):
        embeddings = embeddings.cpu().numpy()

    # 3. تطبيق خفض الأبعاد ثنائي الأبعاد باستخدام PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)

    # 4. رسم المخطط الملون وتحديد مواصفاته الفنية
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=df['cluster_id'], 
        cmap='viridis', 
        alpha=0.6, 
        edgecolors='none', 
        s=10
    )
    
    plt.title('MS MARCO / argsme - 2D Semantic Document Clustering (K-Means)', fontsize=13, pad=15)
    plt.xlabel('PCA Component 1', fontsize=11)
    plt.ylabel('PCA Component 2', fontsize=11)
    plt.colorbar(scatter, label='Semantic Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)

    # حفظ الصورة في مجلد المشروع
    plt.savefig('clustering_chart.png', dpi=300, bbox_inches='tight')
    print("✅ تم بنجاح توليد وحفظ مخطط المجموعات الدلالية باسم: clustering_chart.png")

except Exception as e:
    print(f"❌ حدث خطأ أثناء توليد الرسم البياني: {e}")