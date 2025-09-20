"""
Complete CST Transformer Implementation
Integrates CST module with standard transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math

from cst_module import CSTModule


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations
        Q = self.w_q(query)  # [batch_size, seq_len, d_model]
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = self.w_o(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class OutputHead(nn.Module):
    """Output head for different tasks"""
    
    def __init__(self, config, task_type: str = 'mlm'):
        super().__init__()
        self.task_type = task_type
        self.d_model = config.d_model
        
        if task_type == 'mlm':  # Masked Language Modeling
            self.mlm_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.vocab_size)
            )
        elif task_type == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.d_model // 2, config.num_labels)
            )
        elif task_type == 'generation':
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, 
                pooling_strategy: str = 'mean') -> torch.Tensor:
        
        if self.task_type == 'mlm':
            return self.mlm_head(hidden_states)
        elif self.task_type == 'classification':
            # Pool the sequence representation
            if pooling_strategy == 'cls':
                pooled = hidden_states[:, 0]  # Use [CLS] token
            elif pooling_strategy == 'mean':
                pooled = hidden_states.mean(dim=1)
            elif pooling_strategy == 'max':
                pooled = hidden_states.max(dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
            
            return self.classifier(pooled)
        elif self.task_type == 'generation':
            return self.lm_head(hidden_states)


class CSTransformer(nn.Module):
    """
    Complete CST-enabled Transformer model
    Replaces standard embedding lookup with CST module
    """
    
    def __init__(self, config, task_type: str = 'mlm'):
        super().__init__()
        self.config = config
        self.task_type = task_type
        
        # CST Module (replaces standard token embedding)
        self.cst_module = CSTModule(config)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.d_model, 
            config.max_sequence_length, 
            config.dropout
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_model * 4,  # Standard scaling
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        # Output head
        self.output_head = OutputHead(config, task_type)
        
        # Layer normalization before output
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights using standard transformer initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_attention_mask(self, input_ids: torch.Tensor, 
                            padding_token_id: int = 0) -> torch.Tensor:
        """Create attention mask from input IDs"""
        return (input_ids != padding_token_id).float()
    
    def forward(self, 
                input_ids: torch.Tensor,
                context_data: Dict[str, Any],
                attention_mask: Optional[torch.Tensor] = None,
                fragment_chars: Optional[torch.Tensor] = None,
                context_chars: Optional[torch.Tensor] = None,
                fragment_frequencies: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CST Transformer
        
        Args:
            input_ids: [batch_size, seq_len] - Token IDs
            context_data: Dictionary of contextual information for CST
            attention_mask: [batch_size, seq_len] - Attention mask
            fragment_chars: [batch_size, seq_len, char_len] - Character data
            context_chars: [batch_size, seq_len, context_len] - Context characters
            fragment_frequencies: [batch_size, seq_len] - Token frequencies
            labels: [batch_size, seq_len] or [batch_size] - Labels for loss computation
        """
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # CST Module - Generate contextual spectrum vectors
        spectrum_vectors = self.cst_module(
            text_fragments=input_ids,
            context_data=context_data,
            fragment_chars=fragment_chars,
            context_chars=context_chars,
            fragment_frequencies=fragment_frequencies
        )
        
        # Add positional encoding
        positioned_vectors = self.pos_encoding(spectrum_vectors)
        
        # Process through transformer layers
        hidden_states = positioned_vectors
        attention_weights = []
        
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, mask=attention_mask)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Output head
        if self.task_type == 'classification':
            logits = self.output_head(hidden_states, pooling_strategy='mean')
        else:
            logits = self.output_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.task_type == 'mlm':
                # Masked Language Modeling loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            elif self.task_type == 'classification':
                loss = F.cross_entropy(logits, labels)
            elif self.task_type == 'generation':
                # Shift labels for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'loss': loss,
            'cst_stats': self.cst_module.get_performance_stats()
        }
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 context_data: Dict[str, Any],
                 max_length: int = 50,
                 temperature: float = 1.0,
                 do_sample: bool = True,
                 top_k: int = 50,
                 top_p: float = 0.95) -> torch.Tensor:
        """
        Generate text using the CST Transformer
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(generated, context_data)
                logits = outputs['logits']
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end-of-sequence tokens
                if (next_token == 2).all():  # Assuming 2 is EOS token
                    break
        
        return generated
    
    def get_embeddings(self, input_ids: torch.Tensor, context_data: Dict[str, Any]) -> torch.Tensor:
        """Get CST embeddings without full forward pass"""
        return self.cst_module(input_ids, context_data)
    
    def enable_cst_profiling(self, enable: bool = True):
        """Enable CST performance profiling"""
        self.cst_module.enable_profiling_mode(enable)
    
    def get_cst_stats(self) -> Dict[str, Any]:
        """Get CST performance statistics"""
        return self.cst_module.get_performance_stats()
    
    def save_pretrained(self, save_directory: str):
        """Save the model (simplified version)"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save config
        config_dict = {
            'd_model': self.config.d_model,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'vocab_size': self.config.vocab_size,
            'max_sequence_length': self.config.max_sequence_length,
            'task_type': self.task_type
        }
        
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save CST-specific data
        self.cst_module.save_ambiguous_vocab(
            os.path.join(save_directory, 'ambiguous_vocab.json')
        )


def test_cst_transformer():
    """Test the complete CST Transformer"""
    from config import CSTConfig
    
    config = CSTConfig()
    config.num_layers = 6  # Smaller for testing
    config.ambiguous_word_ids = [1, 5, 10, 15, 20]
    
    # Test different task types
    for task_type in ['mlm', 'classification']:
        print(f"\nTesting CST Transformer for {task_type}...")
        
        if task_type == 'classification':
            config.num_labels = 5
        
        model = CSTransformer(config, task_type=task_type)
        model.enable_cst_profiling(True)
        
        batch_size = 2
        seq_len = 16
        
        # Sample data
        input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
        
        context_data = {
            'document_embedding': torch.randn(batch_size, config.raw_doc_dim),
            'metadata': {
                'author': torch.randint(0, config.num_authors, (batch_size,)),
                'domain': torch.randint(0, config.num_domains, (batch_size,)),
            }
        }
        
        # Create labels
        if task_type == 'mlm':
            labels = input_ids.clone()
            # Mask some tokens
            mask_positions = torch.rand(batch_size, seq_len) < 0.15
            labels[~mask_positions] = -100
        else:
            labels = torch.randint(0, 5, (batch_size,))
        
        # Forward pass
        outputs = model(input_ids, context_data, labels=labels)
        
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output logits shape: {outputs['logits'].shape}")
        print(f"  Loss: {outputs['loss'].item():.4f}")
        
        # Check CST statistics
        cst_stats = outputs['cst_stats']
        print(f"  CST Cache hit rate: {cst_stats.get('hit_rate', 0):.2%}")
        print(f"  Ambiguous tokens: {cst_stats.get('ambiguous_tokens', 0)}")
        
        # Test generation (only for generation task)
        if task_type == 'generation':
            generated = model.generate(
                input_ids[:1], 
                {k: v[:1] if isinstance(v, torch.Tensor) else v for k, v in context_data.items()},
                max_length=10
            )
            print(f"  Generated sequence shape: {generated.shape}")
    
    print("\nCST Transformer tests passed!")


if __name__ == "__main__":
    test_cst_transformer()