import streamlit as st

def show_results(resume_text, results):

    # Resume text
    with st.expander("📄 View Extracted Resume Text"):
        st.write(resume_text)

    st.divider()

    # Job matches
    st.subheader("🔥 Top Job Matches")
    st.dataframe(
        results[["job title", "score"]],
        use_container_width=True,
        hide_index=True
    )
