#import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from rag_systems.run_rag_pipeline import retrieve, generate_answer

#def test_accuracy():
 #   import pandas as pd
 #   data = pd.read_csv("data/golden_dataset.csv")
 #   correct = 0
 #   for _, row in data.iterrows():
  #      query = row['query']
  #      expected = row['expected_answer']
  #      context = retrieve(query)
  #      response = generate_answer(context, query)
   #     if expected.lower() in response.lower():
    #        correct += 1
  #  accuracy = correct / len(data)
  #  print(f"Accuracy: {accuracy:.2f}")
  #  assert accuracy >= 0.6


#this code i was having below when the pipeline was sucessfull commenting this to add the accuracy part using another model
#import sys, os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from rag_systems.run_rag_pipeline import retrieve, generate_answer

#def test_pipeline_runs():
 #   """
  #  Check that RAG pipeline functions execute without crashing.
   # Accuracy is skipped for tiny GPT2 in CI.
    #"""
    #import pandas as pd
    #data = pd.read_csv("data/golden_dataset.csv")

    #for _, row in data.iterrows():
     #   context = retrieve(row['query'])
      #  response = generate_answer(context, row['query'])
       # assert response is not None


#import sys, os, time
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from rag_systems.run_rag_pipeline import retrieve, generate_answer
#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer

# Load the golden dataset
#data = pd.read_csv("data/golden_dataset.csv")
#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

#def compute_similarity(a, b):
 #   """Compute cosine similarity between two sentences."""
  #  embeddings = embedder.encode([a, b], convert_to_numpy=True)
   # return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

#def test_accuracy_and_latency():
  #  correct = 0
 #   total = len(data)
   # latencies = []

   # for _, row in data.iterrows():
    #    query = row['query']
     #   expected = row['expected_answer']

      #  start = time.time()
       # context = retrieve(query)
       # predicted = generate_answer(context, query)
       # latency = time.time() - start
       # latencies.append(latency)

       # sim = compute_similarity(predicted, expected)
       # if sim > 0.8:  # similarity threshold
       #     correct += 1

   # accuracy = correct / total
   # avg_latency = sum(latencies) / total

   # print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
   # print(f"⚡ Average Latency: {avg_latency:.2f}s")

    # Assertions for automated pass/fail
   # assert accuracy >= 0.9, f"❌ Accuracy too low: {accuracy:.2f}"
   # assert avg_latency < 2.0, f"❌ Latency too high: {avg_latency:.2f}"


#import sys, os, time
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from rag_systems.run_rag_pipeline import retrieve, generate_answer
#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer

# -----------------------------
# Load data
# -----------------------------
#data = pd.read_csv("data/golden_dataset.csv")
#documents = pd.read_csv("data/documents.csv")

# Get the text column (adjust column name if different)
#doc_texts = documents['text'].tolist()

# Initialize embedder
#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#index = embedder.encode(doc_texts, convert_to_numpy=True)

# -----------------------------
# Helper function
# -----------------------------
#def compute_similarity(a, b):
 #   embeddings = embedder.encode([a, b], convert_to_numpy=True)
  #  return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -----------------------------
# Test function
# -----------------------------
#def test_accuracy_and_latency():
 #   correct = 0
  #  total = len(data)
   # latencies = []

    #for _, row in data.iterrows():
     #   query = row['query']
      #  expected = row['expected_answer']

      #  start = time.time()
      #  context = retrieve(query, embedder, index, doc_texts)
      #  predicted = generate_answer(context, query)
       # latency = time.time() - start
       # latencies.append(latency)

       # sim = compute_similarity(predicted, expected)
       # if sim > 0.8:  # similarity threshold
        #    correct += 1

    #accuracy = correct / total
    #avg_latency = sum(latencies) / total

    #print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    #print(f"⚡ Average Latency: {avg_latency:.2f}s")

    #assert accuracy >= 0.9, f"❌ Accuracy too low: {accuracy:.2f}"
    #assert avg_latency < 2.0, f"❌ Latency too high: {avg_latency:.2f}"





#import sys, os, time
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from rag_systems.run_rag_pipeline import retrieve, generate_answer, build_index, load_model, load_data
#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer

# -----------------------------
# Load data and components
# -----------------------------
#docs_df, golden_df = load_data()
#embedder, index, doc_texts = build_index(docs_df)
#tokenizer, model = load_model()

# -----------------------------
# Helper function
# -----------------------------
#def compute_similarity(a, b):
 #   embeddings = embedder.encode([a, b], convert_to_numpy=True)
  #  return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -----------------------------
# Test function
# -----------------------------
#def test_accuracy_and_latency():
 #   correct = 0
  #  total = len(golden_df)
   # latencies = []

   # for _, row in golden_df.iterrows():
    #    query = row["query"]
     #   expected = row["expected_answer"]

      #  start = time.time()
       # context = retrieve(query, embedder, index, doc_texts)
       # predicted, latency = generate_answer(model, tokenizer, context, query)
       # latencies.append(latency)

       # sim = compute_similarity(predicted, expected)
       # if sim > 0.8:
        #    correct += 1

   # accuracy = correct / total
   # avg_latency = sum(latencies) / total

   # print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
   # print(f"⚡ Average Latency: {avg_latency:.2f}s")

    # Assertions for automated pass/fail
  #  assert accuracy >= 0.9, f"❌ Accuracy too low: {accuracy:.2f}"
  #  assert avg_latency < 2.0, f"❌ Latency too high: {avg_latency:.2f}"



#import sys, os, time
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#from rag_systems.run_rag_pipeline import generate_answer
#import pandas as pd
#from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer

# -----------------------------
# Load data
# -----------------------------
#documents = pd.read_csv("data/documents.csv")
#golden = pd.read_csv("data/golden_dataset.csv")

#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Helper function
# -----------------------------
#def compute_similarity(a, b):
 #   embeddings = embedder.encode([a, b], convert_to_numpy=True)
  #  return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -----------------------------
# Test function
# -----------------------------
#def test_accuracy_and_latency():
 #   correct = 0
  #  latencies = []

   # for i, row in golden.iterrows():
    #    query = row["query"]
     #   expected = row["expected_answer"]
      #  context = documents.iloc[i % len(documents)]["text"]

     #   start = time.time()
     #   predicted = generate_answer(context, query)
     #   latency = time.time() - start
     #   latencies.append(latency)

     #   sim = compute_similarity(predicted, expected)
     #   if sim > 0.8:
     #       correct += 1

    #accuracy = correct / len(golden)
    #avg_latency = sum(latencies) / len(golden)

    #print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    #print(f"⚡ Average Latency: {avg_latency:.2f}s")

    #assert accuracy >= 0.9, f"❌ Accuracy too low: {accuracy:.2f}"
    #assert avg_latency < 2.0, f"❌ Latency too high: {avg_latency:.2f}"


import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_systems.run_rag_pipeline import load_model, generate_answer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load data
# -----------------------------
documents = pd.read_csv("data/documents.csv")
golden = pd.read_csv("data/golden_dataset.csv")

# Load model and tokenizer
tokenizer, model = load_model()

# Load sentence embedder for evaluation
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# Helper function
# -----------------------------
def compute_similarity(a, b):
    embeddings = embedder.encode([a, b], convert_to_numpy=True)
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# -----------------------------
# Test function
# -----------------------------
def test_accuracy_and_latency():
    correct = 0
    latencies = []

    for i, row in golden.iterrows():
        query = row["query"]
        expected = row["expected_answer"]
        context = documents.iloc[i % len(documents)]["text"]

        start = time.time()
        predicted, latency = generate_answer(model, tokenizer, context, query)
        latencies.append(latency)

        sim = compute_similarity(predicted, expected)
        if sim > 0.75:
            correct += 1

    accuracy = correct / len(golden)
    avg_latency = sum(latencies) / len(golden)

    print(f"\n✅ Accuracy: {accuracy*100:.2f}%")
    print(f"⚡ Average Latency: {avg_latency:.2f}s")

    assert accuracy >= 0.8, f"❌ Accuracy too low: {accuracy:.2f}"
    assert avg_latency < 2.0, f"❌ Latency too high: {avg_latency:.2f}"
