from typing import List, Optional
import os
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from docling.document_converter import DocumentConverter
import torch

class LocalRAGSystem:
    def __init__(
        self,
        pdf_directory: str,
        model_name: str = "mistral",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the RAG system with local components.
        
        Args:
            pdf_directory: Directory containing PDF files
            model_name: Name of the Ollama model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.pdf_dir = Path(pdf_directory)
        self.documents = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize docling converter
        self.converter = DocumentConverter()
        
        # Initialize local LLM through Ollama
        self.llm = Ollama(model=model_name)
        
        # Initialize retriever
        self.retriever = None
    
    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert a PDF file to markdown using docling.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown formatted string
        """
        result = self.converter.convert(pdf_path)
        return result.document.export_to_markdown()
    
    def process_pdfs(self) -> None:
        """Process all PDFs in the specified directory and prepare them for retrieval."""
        if not self.pdf_dir.exists():
            raise ValueError(f"Directory {self.pdf_dir} does not exist")
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.pdf_dir}")
        
        all_docs = []
        for pdf_path in pdf_files:
            try:
                # Convert PDF to markdown using docling
                markdown_text = self.convert_pdf_to_markdown(str(pdf_path))
                
                # Create LangChain documents
                doc = Document(
                    page_content=markdown_text,
                    metadata={
                        "source": pdf_path.name,
                        "format": "markdown"
                    }
                )
                
                # Split documents into chunks
                split_docs = self.text_splitter.split_documents([doc])
                all_docs.extend(split_docs)
                
                print(f"Successfully processed: {pdf_path.name}")
                
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {str(e)}")
                continue
        
        if not all_docs:
            raise ValueError("No documents were successfully processed")
            
        self.documents = all_docs
        
        # Initialize BM25 retriever with processed documents
        self.retriever = BM25Retriever.from_documents(
            self.documents,
            preprocess_func=lambda text: text.lower().split()
        )
        
        print(f"Processed {len(pdf_files)} PDFs into {len(self.documents)} chunks")
        
        # Optionally save markdown versions
        self.save_markdown_versions()
    
    def save_markdown_versions(self, output_dir: Optional[str] = None) -> None:
        """
        Save markdown versions of processed PDFs.
        
        Args:
            output_dir: Optional directory to save markdown files. If None,
                       saves in the same directory as PDFs with .md extension
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = self.pdf_dir
            
        for pdf_path in self.pdf_dir.glob("*.pdf"):
            try:
                markdown_text = self.convert_pdf_to_markdown(str(pdf_path))
                markdown_path = output_path / f"{pdf_path.stem}.md"
                
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                
                print(f"Saved markdown version: {markdown_path}")
                
            except Exception as e:
                print(f"Error saving markdown for {pdf_path.name}: {str(e)}")
    
    def generate_response(
        self,
        query: str,
        k: int = 4,
        max_tokens: int = 512
    ) -> dict:
        """
        Generate a response to a query using RAG.
        
        Args:
            query: User query
            k: Number of relevant documents to retrieve
            max_tokens: Maximum tokens for response generation
            
        Returns:
            dict containing response and source documents
        """
        if not self.retriever:
            raise ValueError("System not initialized. Call process_pdfs() first")
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(query, k=k)
        
        # Construct prompt with context, noting it's markdown formatted
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""Use the following markdown-formatted context to answer the question. 
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        response = self.llm.predict(prompt, max_tokens=max_tokens)
        
        return {
            "response": response,
            "sources": [doc.metadata["source"] for doc in relevant_docs]
        }

def main():
    # Example usage
    pdf_dir = "content"
    
    # Initialize and set up the RAG system
    rag_system = LocalRAGSystem(
        pdf_directory=pdf_dir,
        model_name="mistral"
    )
    
    try:
    
        # Process PDFs
        rag_system.process_pdfs()
        
        # Example query
        query = "who worked in VMware?"
        result = rag_system.generate_response(query)
        
        print(f"\nQuery: {query}")
        print(f"\nResponse: {result['response']}")
        print("\nSources:", ", ".join(result['sources']))
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()