"""
Fragment Encoder for CST
Encodes text fragments with local context using CNN and mini-transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for character sequences"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CharacterCNN(nn.Module):
    """CNN-based character encoder for local pattern recognition"""
    
    def __init__(self, char_embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            # Layer 1: Small receptive field for character n-grams
            nn.Conv1d(char_embed_dim, hidden_dim // 2, kernel_size=3, padding=1),
            # Layer 2: Medium receptive field for morphemes
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            # Layer 3: Large receptive field for word-level patterns
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        ])
        
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ])
        
        self.dropout = nn.Dropout(0.1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, char_embed_dim]
        x = x.transpose(1, 2)  # [batch_size, char_embed_dim, seq_len]
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global max pooling
        x = self.pool(x).squeeze(-1)  # [batch_size, hidden_dim]
        return self.projection(x)


class MiniTransformer(nn.Module):
    """Lightweight transformer for context encoding"""
    
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, seq_len, d_model]
        x = self.pos_encoding(x)
        return self.transformer(x, src_key_padding_mask=mask)


class FragmentEncoder(nn.Module):
    """
    Encodes text fragments with local context using both CNN and Transformer approaches
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Character embeddings
        self.char_embeddings = nn.Embedding(
            config.char_vocab_size, 
            config.char_embed_dim,
            padding_idx=0
        )
        
        # CNN pathway for local pattern recognition
        self.char_cnn = CharacterCNN(
            char_embed_dim=config.char_embed_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim // 2
        )
        
        # Mini-transformer for contextual encoding
        self.context_transformer = MiniTransformer(
            d_model=config.char_embed_dim,
            nhead=4,
            num_layers=2,
            dim_feedforward=config.hidden_dim
        )
        
        # Fusion layers
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=config.char_embed_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output projection
        total_dim = config.hidden_dim // 2 + config.char_embed_dim
        self.output_projection = nn.Sequential(
            nn.Linear(total_dim, config.fragment_encoding_dim),
            nn.LayerNorm(config.fragment_encoding_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Fragment position embeddings
        self.fragment_pos_embedding = nn.Embedding(
            config.max_sequence_length, 
            config.fragment_encoding_dim
        )
        
    def _create_padding_mask(self, char_ids):
        """Create padding mask for character sequences"""
        return char_ids == 0
    
    def forward(self, fragment_chars, context_chars, fragment_positions=None):
        """
        Args:
            fragment_chars: [batch_size, fragment_len] - Character IDs for fragment
            context_chars: [batch_size, context_len] - Character IDs for context
            fragment_positions: [batch_size] - Position of fragment in sequence
        """
        batch_size = fragment_chars.size(0)
        
        # Embed characters
        fragment_embedded = self.char_embeddings(fragment_chars)  # [B, F, D]
        context_embedded = self.char_embeddings(context_chars)    # [B, C, D]
        
        # Create full sequence (context + fragment)
        full_sequence = torch.cat([context_embedded, fragment_embedded], dim=1)  # [B, F+C, D]
        full_mask = self._create_padding_mask(
            torch.cat([context_chars, fragment_chars], dim=1)
        )
        
        # CNN pathway - process full sequence
        cnn_features = self.char_cnn(full_sequence)  # [B, H//2]
        
        # Transformer pathway - contextual encoding
        transformer_output = self.context_transformer(full_sequence, mask=full_mask)
        
        # Extract fragment representation using attention
        fragment_start = context_chars.size(1)
        fragment_repr = transformer_output[:, fragment_start:]  # [B, F, D]
        
        # Attention-based aggregation of fragment tokens
        fragment_aggregated, _ = self.fusion_attention(
            fragment_repr.mean(dim=1, keepdim=True),  # Query: mean of fragment
            fragment_repr,  # Key, Value: all fragment tokens
            fragment_repr
        )
        fragment_aggregated = fragment_aggregated.squeeze(1)  # [B, D]
        
        # Combine CNN and transformer features
        combined_features = torch.cat([cnn_features, fragment_aggregated], dim=-1)
        
        # Final projection
        output = self.output_projection(combined_features)
        
        # Add positional information if provided
        if fragment_positions is not None:
            pos_embeddings = self.fragment_pos_embedding(fragment_positions)
            output = output + pos_embeddings
            
        return output
    
    def encode_batch(self, batch_data):
        """
        Batch encoding with proper handling of variable-length sequences
        
        Args:
            batch_data: List of dicts with keys:
                - 'fragment_chars': tensor of character IDs
                - 'context_chars': tensor of character IDs  
                - 'fragment_position': int position
        """
        # Pad sequences to same length within batch
        fragment_chars = []
        context_chars = []
        fragment_positions = []
        
        max_fragment_len = max(len(item['fragment_chars']) for item in batch_data)
        max_context_len = max(len(item['context_chars']) for item in batch_data)
        
        for item in batch_data:
            # Pad fragment
            frag = item['fragment_chars']
            frag_padded = F.pad(frag, (0, max_fragment_len - len(frag)), value=0)
            fragment_chars.append(frag_padded)
            
            # Pad context
            ctx = item['context_chars']
            ctx_padded = F.pad(ctx, (0, max_context_len - len(ctx)), value=0)
            context_chars.append(ctx_padded)
            
            fragment_positions.append(item['fragment_position'])
        
        fragment_chars = torch.stack(fragment_chars)
        context_chars = torch.stack(context_chars)
        fragment_positions = torch.tensor(fragment_positions, dtype=torch.long)
        
        return self.forward(fragment_chars, context_chars, fragment_positions)


class SubwordFragmentEncoder(nn.Module):
    """Alternative implementation using subword tokenization"""
    
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Subword embeddings
        self.subword_embeddings = nn.Embedding(
            tokenizer.vocab_size,
            config.char_embed_dim,
            padding_idx=tokenizer.pad_token_id
        )
        
        # Context encoder
        self.context_encoder = MiniTransformer(
            d_model=config.char_embed_dim,
            nhead=4,
            num_layers=3,
            dim_feedforward=config.hidden_dim
        )
        
        # Fragment-specific attention
        self.fragment_attention = nn.MultiheadAttention(
            embed_dim=config.char_embed_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.output_projection = nn.Linear(
            config.char_embed_dim, 
            config.fragment_encoding_dim
        )
        
    def forward(self, input_ids, attention_mask, fragment_spans):
        """
        Args:
            input_ids: [batch_size, seq_len] - Tokenized input
            attention_mask: [batch_size, seq_len] - Attention mask
            fragment_spans: [batch_size, 2] - (start, end) indices for fragments
        """
        # Embed tokens
        embeddings = self.subword_embeddings(input_ids)
        
        # Encode full context
        context_encoded = self.context_encoder(embeddings, mask=~attention_mask.bool())
        
        # Extract fragment representations
        batch_size = input_ids.size(0)
        fragment_reprs = []
        
        for i in range(batch_size):
            start, end = fragment_spans[i]
            fragment_tokens = context_encoded[i, start:end+1]  # [frag_len, d_model]
            
            # Use attention to get fragment representation
            fragment_repr, _ = self.fragment_attention(
                fragment_tokens.mean(dim=0, keepdim=True).unsqueeze(0),
                fragment_tokens.unsqueeze(0),
                fragment_tokens.unsqueeze(0)
            )
            fragment_reprs.append(fragment_repr.squeeze())
            
        fragment_reprs = torch.stack(fragment_reprs)
        return self.output_projection(fragment_reprs)
