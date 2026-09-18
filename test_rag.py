import pypdfium2 as pdfium
import os
import numpy as np
from google import genai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    print("❌ Error: GEMINI_API_KEY not found in .env")
    exit(1)

def main():
    print("🚀 Starting Automated QA Benchmark on 'Attention Is All You Need'...")
    
    print("\n[1/3] Extracting Text from PDF...")
    pdf = pdfium.PdfDocument("test_document_short.pdf")
    text = ""
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        text_page = page.get_textpage()
        text += text_page.get_text_range() + "\n"
    
    print(f"✅ Extracted {len(text)} chars of text")
    
    print("\n[2/3] Chunking and Indexing...")
    raw_chunks = text.split("\n\n")
    all_chunks = [c for c in raw_chunks if len(c.strip()) > 50]
    
    # Use TF-IDF for instantaneous local retrieval (no APIs or PyTorch needed)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    print("\n[3/3] Running QA Tests (via Gemini Flash-Lite)...")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    questions = [
        ("Text/Math Test", "What is the exact equation for Scaled Dot-Product Attention?"),
        ("Table Test", "According to the abstract, what BLEU score did they achieve on WMT 2014 English-to-German?"),
        ("Architecture Test", "Describe the architecture of the Transformer model as described in the text.")
    ]
    
    for test_type, query in questions:
        print(f"\n--- {test_type} ---")
        print(f"Q: {query}")
        
        # Search
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-3:][::-1]
        top_chunks = [all_chunks[i] for i in top_indices]
        
        context = "\n\n---\n\n".join(top_chunks)
        
        prompt = f"""You are a highly accurate AI assistant. Answer the user's question based ONLY on the provided context. If it's not in the context, say "I don't know".
Context:
{context}

Question: {query}"""

        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            answer = resp.text
            print(f"A: {answer}")
        except Exception as e:
            print(f"❌ Error getting answer: {str(e)}")

if __name__ == "__main__":
    main()

