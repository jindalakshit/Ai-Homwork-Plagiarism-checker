import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import tempfile
import fitz  
import docx

#  Load all homework files from the 'data' folder
st.set_page_config(page_title="Homework Plagiarism Detector", layout="centered")
st.title("📚 AI Homework Plagiarism Detector")

uploaded_files = st.file_uploader(
    "📂 Upload student homework files (.txt, .pdf, .docx)", 
    type=["txt", "pdf", "docx"], 
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as temp_dir:
        homework_texts = {}
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            ext = uploaded_file.name.lower().split(".")[-1]
            try:
                if ext == "txt":
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                elif ext == "pdf":
                    doc = fitz.open(file_path)
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    doc.close()
                elif ext == "docx":
                    doc = docx.Document(file_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                else:
                    text = ""
                
                homework_texts[uploaded_file.name] = text
            except Exception as e:
                st.error(f"❌ Failed to read {uploaded_file.name}: {e}")

        names = list(homework_texts.keys())
        texts = list(homework_texts.values())

        st.success(f"✅ Loaded {len(names)} homework files!")

        # Generate embeddings using SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(embeddings)

        # Display heatmap
        df_sim = pd.DataFrame(similarity_matrix, index=names, columns=names)
        st.subheader(" Similarity Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_sim, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
        st.pyplot(fig)

        #  Flag and show highly similar pairs
        st.subheader("⚠️ Potential Plagiarism:")
        flagged = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                score = similarity_matrix[i][j]
                if score > 0.85:
                    st.warning(f"{names[i]} and {names[j]} are very similar! Similarity: {score:.2f}")
                    flagged = True

        if not flagged:
            st.success("🎉 No suspicious similarities found.")
