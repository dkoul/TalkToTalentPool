import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List

# Import our RAG system
from rag_system import LocalRAGSystem  

class StreamlitRAGInterface:
    def __init__(self):
        """Initialize the Streamlit interface and RAG system"""
        st.set_page_config(page_title="PDF Question Answering System", layout="wide")
        
        # Initialize session state
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = 'content'

    def save_uploaded_files(self, uploaded_files: List[st.UploadedFile]) -> List[str]:
        """Save uploaded files to temporary directory and return their paths"""
        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            saved_paths.append(file_path)
        return saved_paths

    def render_ui(self):
        """Render the Streamlit user interface"""
        st.title("PDF RAG")
        
        # File Upload Section
        st.header("1. Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        # Process PDFs Button
        if uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                try:
                    # Save uploaded files
                    pdf_paths = self.save_uploaded_files(uploaded_files)
                    
                    # Initialize RAG system
                    st.session_state.rag_system = LocalRAGSystem(
                        pdf_directory=st.session_state.temp_dir,
                        model_name="mistral"
                    )
                    
                    # Process PDFs
                    st.session_state.rag_system.process_pdfs(pdf_paths)
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    
                    st.success(f"Successfully processed {len(uploaded_files)} PDFs!")
                    
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")

        # Display processed files
        if st.session_state.processed_files:
            st.write("Processed Files:")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")

        # Query Section
        st.header("2. Ask Questions")
        query = st.text_area("Enter your question about the PDFs")
        
        if st.button("Submit Question"):
            if not st.session_state.rag_system:
                st.warning("Please upload and process PDFs first!")
            elif not query:
                st.warning("Please enter a question!")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        # Get response from RAG system
                        result = st.session_state.rag_system.generate_response(query)
                        
                        # Display response in a nice format
                        st.subheader("Answer")
                        st.write(result["response"])
                        
                        # Display sources
                        st.subheader("Sources Used")
                        sources = [Path(source).name for source in result["sources"]]
                        source_counts = {}
                        for source in sources:
                            source_counts[source] = source_counts.get(source, 0) + 1
                        
                        for source, count in source_counts.items():
                            st.write(f"- {source} (referenced {count} times)")
                            
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

def main():
    interface = StreamlitRAGInterface()
    interface.render_ui()

if __name__ == "__main__":
    main()