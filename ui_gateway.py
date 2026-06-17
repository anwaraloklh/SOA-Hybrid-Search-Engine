# ui_gateway.py
import gradio as gr
import requests

PREPROCESSING_URL = "http://127.0.0.1:8001"
RETRIEVAL_URL = "http://127.0.0.1:8002"

def search_gateway(query, dataset_choice, hybrid_type, search_mode, k, alpha, k1, b):
    try:
        # 1. استدعاء خدمة المعالجة المسبقة عبر REST API لتبسيط الاستعلام
        res_prep = requests.post(f"{PREPROCESSING_URL}/preprocess", json={"text": query}).json()
        cleaned_query = res_prep["cleaned_text"]
        
        # 2. استدعاء خدمة تحسين الاستعلام عبر REST API عند التفعيل
        final_query = cleaned_query
        query_info_html = ""
        if "Advanced" in search_mode:
            res_expand = requests.post(f"{PREPROCESSING_URL}/expand", json={"text": query}).json()
            expanded_query = res_expand["expanded_text"]
            final_query = requests.post(f"{PREPROCESSING_URL}/preprocess", json={"text": expanded_query}).json()["cleaned_text"]
            query_info_html = f"""
            <div style='background-color: #f1f3f4; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-family: Arial; direction: ltr;'>
                <span style='color: #5f6368; font-size: 13px;'>🔍 REST API Query Expansion Result:</span><br>
                <span style='color: #1a73e8; font-weight: bold; font-size: 15px;'>"{expanded_query}"</span>
            </div>
            """
        
        # 3. استدعاء خدمة البحث والاسترجاع عبر REST API للحصول على النتائج والترتيب
        dataset_name = "msmarco" if "MS MARCO" in dataset_choice else "argsme"
        search_payload = {
            "query": query,
            "cleaned_query": final_query,
            "k": int(k),
            "alpha": float(alpha),
            "k1": float(k1),
            "b": float(b),
            "dataset_name": dataset_name,
            "hybrid_type": "Parallel" if "Parallel" in hybrid_type else "Serial"
        }
        
        res_search = requests.post(f"{RETRIEVAL_URL}/search", json=search_payload).json()
        docs = res_search["docs"]
        scores = res_search["scores"]
        
        # 4. تنسيق وعرض المخرجات كـ HTML
        html_output = query_info_html + "<div style='font-family: Arial; direction: ltr;'>"
        for i, doc in enumerate(docs):
            html_output += f"""
            <div style='border-bottom: 1px solid #e0e0e0; padding: 15px 0;'>
                <div style='color: #1a0dab; font-size: 18px; font-weight: 500;'>#{i+1}. Document ID: {doc['cleaned_doc_id']}</div>
                <div style='color: #006621; font-size: 13px;'>Combined Relevance Score: {scores[i]:.4f}</div>
                <div style='color: #4d4d4d; font-size: 14px; margin-top: 6px;'>{doc['text']}</div>
            </div>
            """
        html_output += "</div>"
        return html_output
        
    except Exception as e:
        return f"<div style='color:red;'>⚠️ Error connecting to SOA services: {e}</div>"

# بناء واجهة Gradio للمشروع كبوابة ويب مستقلة
demo = gr.Interface(
    fn=search_gateway,
    inputs=[
        gr.Textbox(label="🔍 Enter Search Query", placeholder="e.g., global warming debate"),
        gr.Dropdown(choices=["MS MARCO (Passage Retrieval)", "argsme (Argument Retrieval)"], value="argsme (Argument Retrieval)", label="Select Dataset"),
        gr.Radio(choices=["Parallel Hybrid (التفرعي)", "Serial Hybrid (التسلسلي)"], value="Serial Hybrid (التسلسلي)", label="Select Hybrid Type"),
        gr.Radio(choices=["Basic Search", "Advanced Search with Query Expansion"], value="Basic Search", label="Execution Mode"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results (k)"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Alpha Weight"),
        gr.Slider(minimum=0.1, maximum=3.0, value=1.5, step=0.1, label="BM25 Parameter k1"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="BM25 Parameter b")
    ],
    outputs=gr.HTML(label="SOA Search Results"),
    title="🔎 Enterprise SOA Hybrid Search Engine",
    description="This UI acts as an API Gateway, invoking Preprocessing Microservice (Port 8001) and Retrieval Microservice (Port 8002) via HTTP REST APIs."
)

if __name__ == "__main__":
    demo.launch(server_port=8000, inline=False)