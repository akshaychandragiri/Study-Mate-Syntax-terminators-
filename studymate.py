"""
StudyMate: AI-Powered Academic Assistant
A conversational Q&A system for academic PDFs using IBM Granite 3.3-2B Instruct
"""

import os
import tempfile
from typing import List, Tuple
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

class PDFProcessor:
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text += f"\n[Page {page_num + 1}]\n{page_text}"
        doc.close()
        return text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def process_pdfs(self, pdf_files: List[str]) -> Tuple[List[str], int]:
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []

        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            text = self.extract_text_from_pdf(pdf_file)
            self.documents.append({"filename": filename, "text": text})

            file_chunks = self.chunk_text(text)
            for chunk_idx, chunk in enumerate(file_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    "filename": filename,
                    "chunk_id": chunk_idx
                })

        return self.chunks, len(self.documents)


class SemanticSearchEngine:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []

    def build_index(self, chunks: List[str], metadata: List[dict]):
        print(f"Building FAISS index for {len(chunks)} chunks...")
        self.chunks = chunks
        self.chunk_metadata = metadata

        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(f"Index built successfully with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, dict, float]]:
        if self.index is None:
            return []

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append((
                    self.chunks[idx],
                    self.chunk_metadata[idx],
                    float(distance)
                ))

        return results


class GraniteQAModel:
    def __init__(self, model_name: str = "ibm-granite/granite-3.0-2b-instruct"):
        print(f"Loading LLM: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print("Model loaded successfully")

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join([f"Context {i+1}:\n{chunk}" for i, chunk in enumerate(context_chunks)])

        prompt = f"""You are an academic assistant helping students understand their study materials.

Context from the documents:
{context}

Question: {question}

Provide a clear, accurate answer based on the context above. If the context doesn't contain enough information to answer the question, say so clearly.

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()

        return answer


class StudyMateApp:
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.search_engine = SemanticSearchEngine()
        self.qa_model = None
        self.documents_loaded = False

    def load_documents(self, pdf_files) -> str:
        if pdf_files is None or len(pdf_files) == 0:
            return "❌ No PDF files uploaded. Please upload at least one PDF."

        try:
            pdf_paths = [file.name for file in pdf_files]
            chunks, num_docs = self.pdf_processor.process_pdfs(pdf_paths)

            if len(chunks) == 0:
                return "❌ No text extracted from PDFs. Please check your files."

            self.search_engine.build_index(chunks, self.pdf_processor.chunk_metadata)

            if self.qa_model is None:
                self.qa_model = GraniteQAModel()

            self.documents_loaded = True

            return f"✅ Successfully processed {num_docs} document(s) into {len(chunks)} chunks. Ready for questions!"

        except Exception as e:
            return f"❌ Error processing documents: {str(e)}"

    def answer_question(self, question: str) -> Tuple[str, str]:
        if not self.documents_loaded:
            return "❌ Please upload and process documents first.", ""

        if not question or question.strip() == "":
            return "❌ Please enter a question.", ""

        try:
            search_results = self.search_engine.search(question, top_k=3)

            if not search_results:
                return "❌ No relevant content found in the documents.", ""

            context_chunks = [result[0] for result in search_results]

            sources_info = "\n\n📚 **Sources:**\n"
            for i, (chunk, metadata, distance) in enumerate(search_results):
                sources_info += f"\n**Source {i+1}:** {metadata['filename']} (Chunk {metadata['chunk_id']})\n"
                sources_info += f"```\n{chunk[:200]}...\n```\n"

            answer = self.qa_model.generate_answer(question, context_chunks)

            return answer, sources_info

        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", ""

    def create_interface(self):
        with gr.Blocks(title="StudyMate - AI Academic Assistant", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 📚 StudyMate - AI-Powered Academic Assistant

            Upload your study materials (PDFs) and ask questions in natural language.
            StudyMate uses IBM Granite 3.3-2B Instruct to provide accurate, contextual answers.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_upload = gr.File(
                        label="Upload PDF Documents",
                        file_types=[".pdf"],
                        file_count="multiple",
                        type="filepath"
                    )

                    process_btn = gr.Button("📥 Process Documents", variant="primary", size="lg")

                    status_output = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False
                    )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Ask a Question",
                        placeholder="e.g., What are the main concepts discussed in chapter 3?",
                        lines=2
                    )

                    ask_btn = gr.Button("🤔 Get Answer", variant="primary", size="lg")

                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=8,
                        interactive=False
                    )

                    sources_output = gr.Markdown(label="Sources")

            gr.Markdown("""
            ---
            ### 💡 Tips:
            - Upload multiple PDFs for comprehensive coverage
            - Ask specific questions for better answers
            - Questions can be about definitions, explanations, summaries, or comparisons
            """)

            process_btn.click(
                fn=self.load_documents,
                inputs=[pdf_upload],
                outputs=[status_output]
            )

            ask_btn.click(
                fn=self.answer_question,
                inputs=[question_input],
                outputs=[answer_output, sources_output]
            )

            question_input.submit(
                fn=self.answer_question,
                inputs=[question_input],
                outputs=[answer_output, sources_output]
            )

        return interface


def main():
    print("=" * 60)
    print("StudyMate - AI-Powered Academic Assistant")
    print("=" * 60)

    app = StudyMateApp()
    interface = app.create_interface()

    interface.launch(
        share=True,
        debug=True
    )


if __name__ == "__main__":
    main()
