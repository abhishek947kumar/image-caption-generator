"""
Deep Learning Model Architecture for Image Captioning.
Implements the CNN-LSTM Merge and Recurrent Language Model for Flickr8k style caption generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class ImageCaptionModel(nn.Module):
    """
    CNN-LSTM Neural Network for generating captions from visual representations.
    
    Architecture:
    1. Image Feat Projection: Linear(feature_dim -> embed_dim) + LayerNorm + ReLU + Dropout
    2. Text Embedding: Embedding(vocab_size -> embed_dim) + Dropout
    3. Recurrent Language Model: LSTM(embed_dim -> hidden_dim)
    4. Fusion / Decoder Head: Linear(embed_dim + hidden_dim -> hidden_dim) + ReLU + Dropout + Linear(hidden_dim -> vocab_size)
    """
    def __init__(
        self,
        feature_dim: int = 1024,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        vocab_size: int = 5000,
        dropout_p: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # 1. Visual feature projection
        self.image_fc = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        # 2. Text sequence embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout_p)

        # 3. Recurrent LSTM language decoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Optional hidden state initializer from image features
        self.init_h = nn.Linear(embed_dim, hidden_dim)
        self.init_c = nn.Linear(embed_dim, hidden_dim)

        # 4. Multi-modal fusion & Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, vocab_size)
        )

    def init_hidden_states(self, img_embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states from projected image embeddings.
        Args:
            img_embeds: (batch_size, embed_dim)
        Returns:
            h_0: (1, batch_size, hidden_dim)
            c_0: (1, batch_size, hidden_dim)
        """
        h_0 = torch.tanh(self.init_h(img_embeds)).unsqueeze(0)
        c_0 = torch.tanh(self.init_c(img_embeds)).unsqueeze(0)
        return h_0, c_0

    def forward(
        self, 
        image_features: torch.Tensor, 
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training with Teacher Forcing.
        
        Args:
            image_features: (batch_size, feature_dim) extracted by CNN
            captions: (batch_size, seq_len) token IDs starting with <start>
            
        Returns:
            logits: (batch_size, seq_len, vocab_size) prediction probabilities
        """
        batch_size, seq_len = captions.shape
        
        # Project image features -> (batch_size, embed_dim)
        img_embeds = self.image_fc(image_features)
        
        # Embed caption tokens -> (batch_size, seq_len, embed_dim)
        text_embeds = self.embed_dropout(self.embedding(captions))
        
        # Initialize hidden state with visual context
        h_0, c_0 = self.init_hidden_states(img_embeds)
        
        # LSTM pass
        lstm_out, _ = self.lstm(text_embeds, (h_0, c_0))  # (batch_size, seq_len, hidden_dim)
        
        # Broadcast image embeddings across all time steps
        img_expanded = img_embeds.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, embed_dim)
        
        # Concatenate multi-modal features at each time step
        fused = torch.cat([img_expanded, lstm_out], dim=-1)  # (batch_size, seq_len, embed_dim + hidden_dim)
        
        # Output vocabulary logits
        logits = self.classifier(fused)  # (batch_size, seq_len, vocab_size)
        return logits

    def decode_step(
        self,
        img_embeds: torch.Tensor,
        current_token: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single autoregressive decode step for inference (greedy or beam search).
        
        Args:
            img_embeds: (1, embed_dim)
            current_token: (1, 1) tensor containing the previous token ID
            hidden_state: (h, c) from previous step
            
        Returns:
            next_token_logits: (1, vocab_size)
            new_hidden_state: (h, c)
        """
        # Embed single token -> (1, 1, embed_dim)
        token_embed = self.embedding(current_token)
        
        # Run one step of LSTM
        lstm_out, new_hidden = self.lstm(token_embed, hidden_state)  # lstm_out: (1, 1, hidden_dim)
        
        # Concatenate image embedding with lstm output
        fused = torch.cat([img_embeds.unsqueeze(1), lstm_out], dim=-1)  # (1, 1, embed_dim + hidden_dim)
        
        logits = self.classifier(fused).squeeze(1)  # (1, vocab_size)
        return logits, new_hidden

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}
