import torch
import torch.nn as nn
from transformers import AutoModel

class CrossEncoderCSR(nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.d_k = self.encoder.config.hidden_size
        self.total_layers = len(self.encoder.encoder.layer)
        
        # Linear projections for pseudo query and key
        self.W_q = nn.Linear(self.d_k, self.d_k)
        self.W_k = nn.Linear(self.d_k, self.d_k)

    def forward(self, s1_ids, s1_mask, s2_ids, s2_mask, c_ids, c_mask):
        # Encode context and extract [CLS] token embedding
        c = self.encoder(c_ids, c_mask)
        cls_embed = c.last_hidden_state[:, 0, :]
        q_cls = self.W_q(cls_embed)
        
        # Embed sentences using their respective embeddings
        s1_embed = self.encoder.embeddings(s1_ids)
        s2_embed = self.encoder.embeddings(s2_ids)
        
        # Process sentences through the first half of layers
        for i in range(self.total_layers // 2):
            s1_embed = self.encoder.encoder.layer[i](s1_embed, attention_mask=s1_mask.unsqueeze(1).unsqueeze(2))[0]
            s2_embed = self.encoder.encoder.layer[i](s2_embed, attention_mask=s2_mask.unsqueeze(1).unsqueeze(2))[0]
        
        # Project keys and compute attention weights
        k_s1 = self.W_k(s1_embed)
        k_s2 = self.W_k(s2_embed)
        w_s1 = self.get_attention_weights(q_cls, k_s1)
        w_s2 = self.get_attention_weights(q_cls, k_s2)

        # Apply attention weights to sentence embeddings
        s1_embed = (w_s1 + 1).transpose(1, 2) * s1_embed
        s2_embed = (w_s2 + 1).transpose(1, 2) * s2_embed

        # Process sentences through the second half of layers
        for i in range(self.total_layers // 2, self.total_layers):
            s1_embed = self.encoder.encoder.layer[i](s1_embed, attention_mask=s1_mask.unsqueeze(1).unsqueeze(2))[0]
            s2_embed = self.encoder.encoder.layer[i](s2_embed, attention_mask=s2_mask.unsqueeze(1).unsqueeze(2))[0]

        # Average pooling
        s1_hidden = s1_embed.mean(dim=1)
        s2_hidden = s2_embed.mean(dim=1)

        return s1_hidden, s2_hidden

    def get_attention_weights(self, q, k):
        """
        Compute attention weights using cosine similarity.
        """
        scores = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float().to(q.device))
        return torch.nn.functional.softmax(scores, dim=-1)
