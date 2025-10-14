import pandas as pd
from rag_system.model import predict

def test_accuracy():
    data = pd.read_csv("data/golden_dataset.csv")
    correct = 0
    for _, row in data.iterrows():
        query = row['query']
        expected = row['expected_answer']
        response = predict(query)
        if expected.lower() in response.lower():
            correct += 1
    accuracy = correct / len(data)
    print(f"Accuracy: {accuracy:.2f}")
    assert accuracy >= 0.6  # minimum accuracy threshold
