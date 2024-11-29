import torch
from transformers import AutoTokenizer
from model import CrossEncoderCSR
from utils import cosine_similarity

def test_cross_encoder_csr():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Create model
    model = CrossEncoderCSR(model_name="roberta-base")
    
    # Prepare test inputs
    s1 = ["I love playing soccer with my friends."]
    s2 = ["My friends and I enjoy playing football together."]
    c = ["What do you think about soccer as a sport?"]

    # Tokenize inputs
    s1_inputs = tokenizer(s1, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    s2_inputs = tokenizer(s2, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    c_inputs = tokenizer(c, padding="max_length", truncation=True, return_tensors="pt", max_length=128)

    # Forward pass
    with torch.no_grad():
        s1_hidden, s2_hidden = model(
            s1_ids=s1_inputs['input_ids'], 
            s1_mask=s1_inputs['attention_mask'].to(dtype=torch.float),
            s2_ids=s2_inputs['input_ids'], 
            s2_mask=s2_inputs['attention_mask'].to(dtype=torch.float),
            c_ids=c_inputs['input_ids'], 
            c_mask=c_inputs['attention_mask'].to(dtype=torch.float)
        )
    
    # Print results
    print(f"S1 Hidden States Shape: {s1_hidden.shape}")
    print(f"S2 Hidden States Shape: {s2_hidden.shape}")
    
    # Compute cosine similarity
    sim = cosine_similarity(s1_hidden, s2_hidden)
    print(f"Cosine Similarity: {sim.item()}")

if __name__ == "__main__":
    test_cross_encoder_csr()
