# frontend/app_streamlit.py
import streamlit as st
import requests
import pandas as pd
from io import BytesIO

API_URL = ("http://localhost:8000/analyze")


st.title("ClauseWise — Legal Document Analyzer")
st.write("Upload a legal document (PDF / DOCX / TXT) to extract clauses, entities and simplified language.")

uploaded = st.file_uploader("Upload document", type=["pdf","docx","txt"])
if uploaded:
    with st.spinner("Uploading and analyzing..."):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        resp = requests.post(API_URL, files=files)
    if resp.status_code == 200:
        data = resp.json()
        st.success(f"Document classified as **{data['doc_type']}** (score: {data['doc_score']:.2f})")
        clauses = data.get("clauses", [])
        # Show table
        rows = []
        for i,c in enumerate(clauses):
            rows.append({"#": i+1, "Clause (snippet)": c['clause'][:200]+"..." if len(c['clause'])>200 else c['clause'],
                         "Simplified": c['simplified'], "Entities": str(c['entities'])})
        df = pd.DataFrame(rows)
        st.dataframe(df)

        # interactive viewer
        idx = st.number_input("View clause number", min_value=1, max_value=max(1,len(clauses)), value=1)
        sel = clauses[idx-1]
        st.subheader("Original Clause")
        st.write(sel['clause'])
        st.subheader("Simplified")
        st.write(sel['simplified'])
        st.subheader("Entities")
        st.json(sel['entities'])
    else:
        st.error(f"API error: {resp.status_code} {resp.text}")
