import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------------------
# Load documents and create embeddings
# -------------------------------
docs_df = pd.read_csv("data/documents.csv")
doc_texts = docs_df['text'].tolist()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -------------------------------
# Load LLM locally
# -------------------------------
model_name = "mistral7b/Mistral-7B-Instruct-v0"  # Replace with local path if downloaded
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)

# -------------------------------
# Helper functions
# -------------------------------
def retrieve(query, k=3):
    """Retrieve top-k relevant documents using FAISS"""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)
    return " ".join([doc_texts[i] for i in indices[0]])

def generate_answer(context, question, max_tokens=100):
    """Generate answer from LLM using retrieved context"""
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids, max_new_tokens=max_tokens)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.replace(input_text, "").strip()
    return answer

def predict(query):
    """Full RAG pipeline: retrieve + generate"""
    context = retrieve(query, k=3)
    answer = generate_answer(context, query)
    return answer
