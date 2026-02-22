import torch
from transformers import BertTokenizer, BertModel

class BertLayerProbe:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.model.eval()

    def get_hidden_states(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        # tuple of 13 tensors (embedding + 12 layers)
        return outputs.hidden_states

    def get_cls_per_layer(self, sentence):
        hidden_states = self.get_hidden_states(sentence)

        cls_vectors = []
        for layer in hidden_states:
            # layer shape: (batch_size, seq_len, hidden_dim)
            cls_vector = layer[:, 0, :]   # CLS token
            cls_vectors.append(cls_vector)

        return cls_vectors