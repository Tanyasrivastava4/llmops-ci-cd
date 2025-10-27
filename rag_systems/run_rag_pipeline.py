#import os
#import pandas as pd
#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
#from sentence_transformers import SentenceTransformer
#import faiss


# -------------------------------
# Paths
#DATA_DIR = "data"
#OUTPUT_DIR = "outputs"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

#DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.csv")
#GOLDEN_FILE = os.path.join(DATA_DIR, "golden_dataset.csv")
#OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_responses.csv")

# -------------------------------
# Load documents and create embeddings
# -------------------------------
#docs_df = pd.read_csv(DOCUMENTS_FILE)
#doc_texts = docs_df['text'].tolist()

#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

#dimension = doc_embeddings.shape[1]
#index = faiss.IndexFlatL2(dimension)
#index.add(doc_embeddings)

# -------------------------------
# Load Mistral-7B directly from Hugging Face
# -------------------------------
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

#print("Loading tokenizer and model from Hugging Face...")
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModelForCausalLM.from_pretrained(
 #   MODEL_NAME,
  #  torch_dtype=torch.float16,
  #  device_map="auto"  # automatically uses GPU if available
#)

# -------------------------------
# Load golden dataset
# -------------------------------
#golden_df = pd.read_csv(GOLDEN_FILE)

# -------------------------------
# Helper functions
# -------------------------------
#def retrieve(query, k=1):
   
   # query_emb = embedder.encode([query], convert_to_numpy=True)
    #distances, indices = index.search(query_emb, k)
   # retrieved_texts = [doc_texts[i] for i in indices[0]]
  #  return " ".join(retrieved_texts)

#def generate_answer(context, question, max_tokens=100):
    #Generate answer from LLM using retrieved context
 #   input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
  #  input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
  #  outputs = model.generate(input_ids, max_new_tokens=max_tokens)
 #   answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove input text from output if present
 #   answer = answer.replace(input_text, "").strip()
  #  return answer

#def is_correct(predicted, golden):
    #Simple accuracy check (case-insensitive exact match)
#    return predicted.strip().lower() == golden.strip().lower()

# -------------------------------
# Run RAG over golden dataset
# -------------------------------
#responses = []
#correct_count = 0

#for _, row in golden_df.iterrows():
 #   question = row['query']
  #  golden_answer = row['expected_answer']

    # Retrieval
   # context = retrieve(question, k=1)

    # Generate answer
   # predicted_answer = generate_answer(context, question)

    # Accuracy check
   # correct = is_correct(predicted_answer, golden_answer)
 #   if correct:
  #      correct_count += 1

    # Save result
   # responses.append({
    #    "question": question,
     #   "golden_answer": golden_answer,
      #  "predicted_answer": predicted_answer,
     #   "correct": correct,
      #  "context_used": context
   # })

  #  print(f"Q: {question}")
#    print(f"Predicted: {predicted_answer}")
 #   print(f"Golden : {golden_answer}")
#    print(f"Correct? {correct}\n")

# -------------------------------
# Calculate accuracy
# -------------------------------
#total = len(golden_df)
#accuracy = (correct_count / total) * 100
#print(f"✅ Total Accuracy: {accuracy:.2f}% ({correct_count}/{total} correct)")

# -------------------------------
# Save responses to CSV
# -------------------------------
#responses_df = pd.DataFrame(responses)
#responses_df.to_csv(OUTPUT_FILE, index=False)
#print(f"✅ Responses and accuracy saved to {OUTPUT_FILE}")



#this is the code we are having for sucessful deployment of ci/cd pipeline commenting this to adding the accuracy and latency part for the project

#import os
#import pandas as pd
#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
#from sentence_transformers import SentenceTransformer
#import faiss

# -------------------------------
# Paths
#DATA_DIR = "data"
#OUTPUT_DIR = "outputs"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

#DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.csv")
#GOLDEN_FILE = os.path.join(DATA_DIR, "golden_dataset.csv")
#OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_responses.csv")

# -------------------------------
# Load documents and create embeddings
# -------------------------------
#docs_df = pd.read_csv(DOCUMENTS_FILE)
#doc_texts = docs_df['text'].tolist()

#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

#dimension = doc_embeddings.shape[1]
#index = faiss.IndexFlatL2(dimension)
#index.add(doc_embeddings)

# -------------------------------
# Load model (use tiny GPT2 for CI / disk-safe runs)
# -------------------------------
#MODEL_NAME = "sshleifer/tiny-gpt2"  # smaller model for CI

#print("Loading tokenizer and model from Hugging Face...")
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")  # CPU only

# -------------------------------
# Load golden dataset
# -------------------------------
#golden_df = pd.read_csv(GOLDEN_FILE)

# -------------------------------
# Helper functions
# -------------------------------
#def retrieve(query, k=1):
 #   """Retrieve top-k relevant documents using FAISS"""
  #  query_emb = embedder.encode([query], convert_to_numpy=True)
   # distances, indices = index.search(query_emb, k)
   # retrieved_texts = [doc_texts[i] for i in indices[0]]
   # return " ".join(retrieved_texts)

#def generate_answer(context, question, max_tokens=50):
 #   """Generate answer from LLM using retrieved context"""
  #  input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
   # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
   # outputs = model.generate(input_ids, max_new_tokens=max_tokens)
   # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove input text from output if present
   # answer = answer.replace(input_text, "").strip()
   # return answer

# -------------------------------
# Run RAG over golden dataset
# -------------------------------
#responses = []

#for _, row in golden_df.iterrows():
 #   question = row['query']
  #  golden_answer = row['expected_answer']

    # Retrieval
   # context = retrieve(question, k=1)

    # Generate answer
    #predicted_answer = generate_answer(context, question)

    # Save result
   # responses.append({
    #    "question": question,
     #   "golden_answer": golden_answer,
     #   "predicted_answer": predicted_answer,
     #   "context_used": context
    #})

    #print(f"Q: {question}")
   # print(f"Predicted: {predicted_answer}")
   # print(f"Golden : {golden_answer}\n")

# -------------------------------
# Save responses to CSV
# -------------------------------
#responses_df = pd.DataFrame(responses)
#responses_df.to_csv(OUTPUT_FILE, index=False)
#print(f"✅ Responses saved to {OUTPUT_FILE}")



#import os
#import pandas as pd
#import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
#from sentence_transformers import SentenceTransformer
#import faiss
#import time

# -------------------------------
# CONFIG
# -------------------------------
#DATA_DIR = "data"
#OUTPUT_DIR = "outputs"
#os.makedirs(OUTPUT_DIR, exist_ok=True)

#DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.csv")
#GOLDEN_FILE = os.path.join(DATA_DIR, "golden_dataset.csv")
#OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_responses.csv")

# Use a small model for CI/CD to avoid storage and timeout issues
#MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")  # default for CI

# -------------------------------
# LOAD DATA
# -------------------------------
#def load_data():
 #   docs_df = pd.read_csv(DOCUMENTS_FILE)
  #  golden_df = pd.read_csv(GOLDEN_FILE)
   # return docs_df, golden_df


# -------------------------------
# BUILD EMBEDDINGS INDEX
# -------------------------------
#def build_index(docs_df):
 #   doc_texts = docs_df["text"].tolist()
  #  embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   # doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

   # dimension = doc_embeddings.shape[1]
    #index = faiss.IndexFlatL2(dimension)
    #index.add(doc_embeddings)
    #return embedder, index, doc_texts


# -------------------------------
# LOAD MODEL
# -------------------------------
#def load_model():
 #   print(f"🔹 Loading model: {MODEL_NAME}")
  #  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
   # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
    #return tokenizer, model


# -------------------------------
# RAG RETRIEVE + GENERATE
# -------------------------------
#def retrieve(query, embedder, index, doc_texts, k=1):
 #   """Retrieve top-k relevant documents"""
  #  query_emb = embedder.encode([query], convert_to_numpy=True)
   # _, indices = index.search(query_emb, k)
    #return " ".join(doc_texts[i] for i in indices[0])


#def generate_answer(model, tokenizer, context, question, max_tokens=50):
 #   """Generate answer using retrieved context"""
  #  input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
   # input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

   # start_time = time.time()
   # outputs = model.generate(input_ids, max_new_tokens=max_tokens)
   # latency = time.time() - start_time

   # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
   # answer = answer.replace(input_text, "").strip()
   # return answer, latency


# -------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------
#def run_pipeline():
 #   docs_df, golden_df = load_data()
  #  embedder, index, doc_texts = build_index(docs_df)
   # tokenizer, model = load_model()

    #responses = []

   # for _, row in golden_df.iterrows():
    #    question = row["query"]
     #   golden_answer = row["expected_answer"]

#        context = retrieve(question, embedder, index, doc_texts)
 #       predicted_answer, latency = generate_answer(model, tokenizer, context, question)
#
 #       responses.append({
  #          "question": question,
   #         "golden_answer": golden_answer,
    #        "predicted_answer": predicted_answer,
     #       "context_used": context,
      #      "latency": latency,
       # })

        #print(f"Q: {question}")
        #print(f"Predicted: {predicted_answer}")
        #print(f"Golden   : {golden_answer}")
        #print(f"⏱ Latency: {latency:.2f}s\n")

    #responses_df = pd.DataFrame(responses)
    #responses_df.to_csv(OUTPUT_FILE, index=False)
    #print(f"✅ Responses saved to {OUTPUT_FILE}")
    #return responses_df


# -------------------------------
# ENTRY POINT (for manual runs)
# -------------------------------
#if __name__ == "__main__":
 #   run_pipeline()



import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import time

# -------------------------------
# CONFIG
# -------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.csv")
GOLDEN_FILE = os.path.join(DATA_DIR, "golden_dataset.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model_responses.csv")

# Use FLAN-T5 for better reasoning and text generation
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    docs_df = pd.read_csv(DOCUMENTS_FILE)
    golden_df = pd.read_csv(GOLDEN_FILE)
    return docs_df, golden_df


# -------------------------------
# BUILD EMBEDDINGS INDEX
# -------------------------------
def build_index(docs_df):
    doc_texts = docs_df["text"].tolist()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(doc_texts, convert_to_numpy=True)

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return embedder, index, doc_texts


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    print(f"🔹 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu")
    return tokenizer, model


# -------------------------------
# RAG RETRIEVE + GENERATE
# -------------------------------
def retrieve(query, embedder, index, doc_texts, k=1):
    """Retrieve top-k relevant documents"""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_emb, k)
    return " ".join(doc_texts[i] for i in indices[0])


def generate_answer(model, tokenizer, context, question, max_tokens=128):
    """Generate answer using retrieved context (for T5-style model)"""
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(model.device)

    start_time = time.time()
    outputs = model.generate(input_ids, max_new_tokens=max_tokens)
    latency = time.time() - start_time

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip(), latency


# -------------------------------
# MAIN PIPELINE FUNCTION
# -------------------------------
def run_pipeline():
    docs_df, golden_df = load_data()
    embedder, index, doc_texts = build_index(docs_df)
    tokenizer, model = load_model()

    responses = []

    for _, row in golden_df.iterrows():
        question = row["query"]
        golden_answer = row["expected_answer"]

        context = retrieve(question, embedder, index, doc_texts)
        predicted_answer, latency = generate_answer(model, tokenizer, context, question)

        responses.append({
            "question": question,
            "golden_answer": golden_answer,
            "predicted_answer": predicted_answer,
            "context_used": context,
            "latency": latency,
        })

        print(f"Q: {question}")
        print(f"Predicted: {predicted_answer}")
        print(f"Golden   : {golden_answer}")
        print(f"⏱ Latency: {latency:.2f}s\n")

    responses_df = pd.DataFrame(responses)
    responses_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Responses saved to {OUTPUT_FILE}")
    return responses_df


# -------------------------------
# ENTRY POINT (for manual runs)
# -------------------------------
if __name__ == "__main__":
    run_pipeline()
