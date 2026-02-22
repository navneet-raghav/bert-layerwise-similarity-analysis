import torch

def cosine_similarity(vec1, vec2):
    vec1 = vec1.squeeze(0)
    vec2 = vec2.squeeze(0)
    
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()