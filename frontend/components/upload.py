import streamlit as st

def upload_resume():
    st.subheader("📂 Upload Your Resume")

    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["pdf"],
        help="Only PDF files are supported"
    )

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        return "temp.pdf"

    return None
