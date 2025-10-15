import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_systems.run_rag_pipeline import retrieve, generate_answer

def test_accuracy():
    import pandas as pd
    data = pd.read_csv("data/golden_dataset.csv")
    correct = 0
    for _, row in data.iterrows():
        query = row['query']
        expected = row['expected_answer']
        context = retrieve(query)
        response = generate_answer(context, query)
        if expected.lower() in response.lower():
            correct += 1
    accuracy = correct / len(data)
    print(f"Accuracy: {accuracy:.2f}")
    assert accuracy >= 0.6

