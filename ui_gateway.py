# ui_gateway.py
import gradio as gr
import requests
import os

PREPROCESSING_URL = "http://127.0.0.1:8001"
RETRIEVAL_URL = "http://127.0.0.1:8002"
EVALUATION_URL = "http://127.0.0.1:8003"

def search_gateway(query, dataset_choice, hybrid_type, search_mode, cluster_choice, k, alpha, k1, b):
    try:
        # 1. استدعاء خدمة المعالجة المسبقة عبر REST API
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
        
        # إعداد قيمة الكلوستر لتمريرها
        cluster_filter = "All"
        if "Topic" in cluster_choice:
            cluster_filter = cluster_choice.split(" ")[0] + " " + cluster_choice.split(" ")[1]
            
        # خريطة تبديل أسماء الخوارزميات للتوافق مع الـ API
        api_hybrid_type = "Serial"
        if "TF-IDF" in hybrid_type:
            api_hybrid_type = "TF-IDF"
        elif "BM25" in hybrid_type:
            api_hybrid_type = "BM25"
        elif "BERT" in hybrid_type:
            api_hybrid_type = "BERT"
        elif "Parallel" in hybrid_type:
            api_hybrid_type = "Parallel"
            
        # 3. استدعاء خدمة البحث والاسترجاع عبر REST API
        dataset_name = "msmarco" if "MS MARCO" in dataset_choice else "argsme"
        search_payload = {
            "query": query,
            "cleaned_query": final_query,
            "k": int(k),
            "alpha": float(alpha),
            "k1": float(k1),
            "b": float(b),
            "dataset_name": dataset_name,
            "hybrid_type": api_hybrid_type,
            "cluster_filter": cluster_filter
        }
        
        res_search = requests.post(f"{RETRIEVAL_URL}/search", json=search_payload).json()
        docs = res_search["docs"]
        scores = res_search["scores"]
        
        # 4. تنسيق وعرض المخرجات كـ HTML
        html_output = query_info_html + "<div style='font-family: Arial; direction: ltr;'>"
        for i, doc in enumerate(docs):
            topic_num = int(doc['cluster_id']) + 1
            html_output += f"""
            <div style='border-bottom: 1px solid #e0e0e0; padding: 15px 0;'>
                <div style='color: #1a0dab; font-size: 18px; font-weight: 500;'>#{i+1}. Document ID: {doc['cleaned_doc_id']}</div>
                <div style='color: #006621; font-size: 13px; margin-top: 2px;'>
                    Combined Relevance Score: {scores[i]:.4f} | <span style='background-color: #e8f0fe; color: #1a73e8; padding: 2px 6px; border-radius: 3px; font-weight: bold;'>Semantic Group: Topic {topic_num}</span>
                </div>
                <div style='color: #4d4d4d; font-size: 14px; margin-top: 6px;'>{doc['text']}</div>
            </div>
            """
        html_output += "</div>"
        return html_output
        
    except Exception as e:
        return f"<div style='color:red;'>⚠️ Error connecting to SOA services: {e}</div>"


def run_evaluation_gateway(dataset_choice, hybrid_type, eval_mode, alpha, k1, b):
    try:
        dataset_name = "msmarco" if "MS MARCO" in dataset_choice else "argsme"
        
        api_hybrid_type = "Serial"
        if "TF-IDF" in hybrid_type:
            api_hybrid_type = "TF-IDF"
        elif "BM25" in hybrid_type:
            api_hybrid_type = "BM25"
        elif "BERT" in hybrid_type:
            api_hybrid_type = "BERT"
        elif "Parallel" in hybrid_type:
            api_hybrid_type = "Parallel"

        # تحديد هل نقيم قبل أم بعد ميزة التوسيع
        use_query_expansion = True if "Advanced" in eval_mode else False

        eval_payload = {
            "dataset_name": dataset_name,
            "hybrid_type": api_hybrid_type,
            "alpha": float(alpha),
            "k1": float(k1),
            "b": float(b),
            "k": 10,
            "use_query_expansion": use_query_expansion
        }
        
        print(f"🔄 جاري إرسال طلب التقييم لخدمة التقييم (Port 8003)...")
        res_eval = requests.post(f"{EVALUATION_URL}/evaluate", json=eval_payload).json()
        
        # تنسيق مخرجات التقييم كجدول HTML فخم ومطابق للتسليم
        html_table = f"""
        <div style='font-family: Arial, sans-serif; text-align: center; max-width: 600px; margin: 0 auto;'>
            <h3 style='color: #1a73e8; margin-bottom: 15px;'>🏆 نتائج التقييم العلمي للمجموعة ({dataset_choice}) 🏆</h3>
            <table style='width: 100%; border-collapse: collapse; margin-top: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.2);'>
                <thead>
                    <tr style='background-color: #1a73e8; color: white;'>
                        <th style='padding: 12px; border: 1px solid #ddd; text-align: center;'>المقياس العلمي (Metric)</th>
                        <th style='padding: 12px; border: 1px solid #ddd; text-align: center;'>القيمة المحسوبة (Value)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style='padding: 12px; border: 1px solid #ddd; font-weight: bold;'>Precision@10</td>
                        <td style='padding: 12px; border: 1px solid #ddd; color: #2e8b57; font-weight: bold; font-size: 16px;'>{res_eval.get("Precision_at_10", 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border: 1px solid #ddd; font-weight: bold;'>Recall@10</td>
                        <td style='padding: 12px; border: 1px solid #ddd; color: #2e8b57; font-weight: bold; font-size: 16px;'>{res_eval.get("Recall_at_10", 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border: 1px solid #ddd; font-weight: bold;'>MAP (Mean Average Precision)</td>
                        <td style='padding: 12px; border: 1px solid #ddd; color: #2e8b57; font-weight: bold; font-size: 16px;'>{res_eval.get("MAP_at_10", 0.0):.4f}</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px; border: 1px solid #ddd; font-weight: bold;'>nDCG@10</td>
                        <td style='padding: 12px; border: 1px solid #ddd; color: #2e8b57; font-weight: bold; font-size: 16px;'>{res_eval.get("nDCG_at_10", 0.0):.4f}</td>
                    </tr>
                </tbody>
            </table>
            <p style='color: #5f6368; font-size: 12px; margin-top: 15px;'>* تم حساب المقاييس بالتواصل الفعلي الآلي بين خدمة التقييم (8003) وخدمة البحث (8002) عبر REST APIs.</p>
        </div>
        """
        return html_table
    except Exception as e:
        return f"<div style='color:red; text-align:center;'>⚠️ حدث خطأ أثناء الاتصال بخدمة التقييم: {e}</div>"

# دالة ذكية لعرض شارت الكلوسترينغ المحفوظ في المجلد
def get_clustering_chart():
    chart_path = 'clustering_chart.png'
    if os.path.exists(chart_path):
        return chart_path
    return None


# 3. بناء الواجهة الرسومية الشاملة والمقسمة إلى تبويبات (Gradio Blocks)
with gr.Blocks() as demo:
    gr.Markdown("# 🔎 Enterprise SOA Hybrid Search Engine")
    gr.Markdown("This distributed system uses independent microservices on Ports 8001, 8002, and 8003 communicating via REST APIs.")
    
    # التبويب الأول: بوابة البحث التفاعلي
    with gr.Tab("🔍 Search Portal (بوابة البحث)"):
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(label="Enter Search Query", placeholder="e.g., global warming debate")
                dataset_choice = gr.Dropdown(choices=["MS MARCO (Passage Retrieval)", "argsme (Argument Retrieval)"], value="argsme (Argument Retrieval)", label="Select Dataset")
                
                # تم تحويل خيار الراديو إلى قائمة منسدلة ذكية تتيح اختيار أي من النماذج الفردية أو الهجينة
                hybrid_type = gr.Dropdown(
                    choices=["TF-IDF (VSM)", "BM25 Only", "BERT Only (Semantic)", "Parallel Hybrid (التفرعي)", "Serial Hybrid (التسلسلي)"],
                    value="Serial Hybrid (التسلسلي)",
                    label="Select Search Method / Model (نموذج التمثيل والبحث)"
                )
                
                search_mode = gr.Radio(choices=["Basic Search", "Advanced Search with Query Expansion"], value="Basic Search", label="Execution Mode")
                cluster_choice = gr.Dropdown(choices=["All (البحث العام بدون تصفية)", "Topic 1 (المجموعة الأولى)", "Topic 2 (المجموعة الثانية)", "Topic 3 (المجموعة الثالثة)", "Topic 4 (المجموعة الرابعة)", "Topic 5 (المجموعة الخامسة)"], value="All (البحث العام بدون تصفية)", label="Filter by Semantic Cluster")
                k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Results (k)")
                alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Alpha Weight")
                k1_slider = gr.Slider(minimum=0.1, maximum=3.0, value=1.5, step=0.1, label="BM25 Parameter k1")
                b_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="BM25 Parameter b")
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")
            with gr.Column(scale=2):
                search_output = gr.HTML(label="Search Results")
                
        submit_btn.click(
            fn=search_gateway,
            inputs=[query_input, dataset_choice, hybrid_type, search_mode, cluster_choice, k_slider, alpha_slider, k1_slider, b_slider],
            outputs=search_output
        )
        clear_btn.click(lambda: ("", "All (البحث العام بدون تصفية)", ""), outputs=[query_input, cluster_choice, search_output])
        
    # التبويب الثاني: لوحة تحكم التقييم العلمي الموزع (المنفذ 8003)
    with gr.Tab("📊 Live Evaluation Dashboard (تقييم النظام)"):
        gr.Markdown("### Run System-Wide Evaluation directly using the Evaluation Microservice (Port 8003)")
        gr.Markdown("This dashboard allows the evaluator to test the search engine over the standard test collections on-the-fly.")
        
        with gr.Row():
            with gr.Column():
                eval_dataset = gr.Dropdown(choices=["argsme (Argument Retrieval)"], value="argsme (Argument Retrieval)", label="Select Dataset to Evaluate")
                eval_hybrid_type = gr.Dropdown(
                    choices=["TF-IDF (VSM)", "BM25 Only", "BERT Only (Semantic)", "Parallel Hybrid (التفرعي)", "Serial Hybrid (التسلسلي)"],
                    value="Serial Hybrid (التسلسلي)",
                    label="Select Search Method to Evaluate"
                )
                
                # إضافة خيار التقييم قبل وبعد تطبيق الميزات الإضافية (توسيع الاستعلام)
                eval_mode = gr.Radio(
                    choices=["Basic Evaluation (قبل تطبيق ميزة التوسيع)", "Advanced Evaluation with Query Expansion (بعد تطبيق ميزة التوسيع)"],
                    value="Basic Evaluation (قبل تطبيق ميزة التوسيع)",
                    label="Evaluation Mode (نمط التقييم - قبل وبعد الميزات)"
                )
                
                eval_alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Alpha Weight")
                eval_k1 = gr.Slider(minimum=0.1, maximum=3.0, value=1.5, step=0.1, label="BM25 Parameter k1")
                eval_b = gr.Slider(minimum=0.0, maximum=1.0, value=0.75, step=0.05, label="BM25 Parameter b")
                eval_btn = gr.Button("Run Benchmark (بدء حساب مقاييس التقييم)", variant="primary")
            with gr.Column():
                eval_output = gr.HTML(label="Evaluation Metrics Table")
                
        eval_btn.click(
            fn=run_evaluation_gateway,
            inputs=[eval_dataset, eval_hybrid_type, eval_mode, eval_alpha, eval_k1, eval_b],
            outputs=eval_output
        )
        
    # التبويب الثالث: عرض شارت الكلوسترينغ والمجموعات الدلالية الملون
    with gr.Tab("🖼️ Semantic Clusters Chart (مخطط المجموعات)"):
        gr.Markdown("### 📊 2D Semantic Document Clustering Visualization (K-Means)")
        gr.Markdown("This visualization shows the SBERT document embeddings reduced to 2D space using PCA. Documents are clustered into 5 distinct semantic topics using K-Means clustering.")
        
        with gr.Row():
            with gr.Column():
                show_chart_btn = gr.Button("Show Clustering Chart (عرض المخطط الدلالي)", variant="primary")
            with gr.Column():
                chart_image_output = gr.Image(label="K-Means Clusters Scatter Plot", type="filepath")
                
        show_chart_btn.click(
            fn=get_clustering_chart,
            inputs=[],
            outputs=chart_image_output
        )

if __name__ == "__main__":
    demo.launch(server_port=8000)