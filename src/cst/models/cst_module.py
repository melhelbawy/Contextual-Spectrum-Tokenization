"""
Core CST Module Implementation
Main module that orchestrates fragment encoding, information fusion, and caching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
from collections import OrderedDict
import time

from fragment_encoder import FragmentEncoder
from information_fuser import InformationFuser


class LRUCache:
    """Simple LRU cache implementation for embedding caching"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key].clone()  # Clone to avoid in-place modifications
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: torch.Tensor):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
        self.cache[key] = value.clone().detach()
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'capacity': self.capacity
        }


class AmbiguityClassifier(nn.Module):
    """Determines whether dynamic processing is needed for each fragment"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Pre-computed ambiguous word vocabulary (loaded during training)
        self.register_buffer(
            'ambiguous_vocab', 
            torch.tensor(config.ambiguous_word_ids if config.ambiguous_word_ids else [])
        )
        
        # Context-based ambiguity classifier
        context_input_dim = config.fragment_encoding_dim + config.context_feature_dim
        self.context_classifier = nn.Sequential(
            nn.Linear(context_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Frequency-based classifier (learns from data)
        self.frequency_classifier = nn.Sequential(
            nn.Linear(1, 32),  # Input: log frequency
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Combination weights
        self.combination_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3]))  # vocab, context, frequency
        self.ambiguity_threshold = config.ambiguity_threshold
        
    def forward(self, 
                fragment_ids: torch.Tensor, 
                context_features: torch.Tensor,
                fragment_frequencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Determine ambiguity for each fragment
        
        Args:
            fragment_ids: [batch_size] - Fragment token IDs
            context_features: [batch_size, context_feature_dim] - Context features
            fragment_frequencies: [batch_size] - Log frequencies of fragments
        """
        batch_size = fragment_ids.size(0)
        ambiguity_scores = torch.zeros(batch_size, device=fragment_ids.device)
        
        # 1. Vocabulary-based ambiguity
        if len(self.ambiguous_vocab) > 0:
            vocab_ambiguous = torch.isin(fragment_ids, self.ambiguous_vocab).float()
            ambiguity_scores += self.combination_weights[0] * vocab_ambiguous
        
        # 2. Context-based ambiguity  
        if context_features.size(1) >= self.config.context_feature_dim:
            # Pad fragment encoding to match expected dimension
            fragment_encoding = torch.zeros(batch_size, self.config.fragment_encoding_dim, 
                                          device=fragment_ids.device)
            combined_features = torch.cat([fragment_encoding, context_features[:, :self.config.context_feature_dim]], dim=1)
            context_scores = self.context_classifier(combined_features).squeeze(-1)
            ambiguity_scores += self.combination_weights[1] * context_scores
        
        # 3. Frequency-based ambiguity (high frequency words are more likely ambiguous)
        if fragment_frequencies is not None:
            freq_scores = self.frequency_classifier(fragment_frequencies.unsqueeze(-1)).squeeze(-1)
            ambiguity_scores += self.combination_weights[2] * freq_scores
        
        # Return binary decisions
        return ambiguity_scores > self.ambiguity_threshold
    
    def update_ambiguous_vocab(self, new_ambiguous_ids: List[int]):
        """Update the ambiguous vocabulary during training"""
        self.ambiguous_vocab = torch.tensor(new_ambiguous_ids, device=self.ambiguous_vocab.device)


class ProjectionHead(nn.Module):
    """Projects fused representation to transformer embedding dimension"""
    
    def __init__(self, config):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(config.fused_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Tanh(),  # Bounded output for stability
            nn.Dropout(0.1)
        )
        
        # Residual connection option
        self.use_residual = config.fused_dim == config.d_model
        if not self.use_residual and hasattr(config, 'enable_projection_residual'):
            self.residual_proj = nn.Linear(config.fused_dim, config.d_model)
            self.use_residual = config.enable_projection_residual
        
    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        output = self.projection(fused_representation)
        
        if self.use_residual:
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(fused_representation)
            else:
                residual = fused_representation
            output = output + residual
            
        return output


class CSTModule(nn.Module):
    """
    Main Contextual Spectrum Tokenization Module
    
    Integrates fragment encoding, information fusion, ambiguity detection, and caching
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core components
        self.fragment_encoder = FragmentEncoder(config)
        self.information_fuser = InformationFuser(config)
        self.projection_head = ProjectionHead(config)
        self.ambiguity_classifier = AmbiguityClassifier(config)
        
        # Static embeddings fallback
        self.static_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Initialize static embeddings with reasonable values
        nn.init.normal_(self.static_embeddings.weight, mean=0.0, std=0.02)
        
        # Caching system
        self.cache = LRUCache(config.cache_size)
        
        # Performance tracking
        self.enable_profiling = False
        self.profile_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'ambiguous_tokens': 0,
            'static_tokens': 0,
            'total_forward_time': 0.0,
            'num_forward_calls': 0
        }
        
    def _compute_cache_key(self, fragment_data: Dict[str, Any], context_data: Dict[str, Any]) -> str:
        """Compute a hash key for caching"""
        # Create a simplified representation for hashing
        key_components = {
            'fragment_id': fragment_data.get('fragment_id', '').item() if torch.is_tensor(fragment_data.get('fragment_id')) else str(fragment_data.get('fragment_id', '')),
            'context_hash': self._hash_context(context_data)
        }
        
        key_string = json.dumps(key_components, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _hash_context(self, context_data: Dict[str, Any]) -> str:
        """Create a hash of context data for caching"""
        context_summary = {}
        
        for key, value in context_data.items():
            if isinstance(value, torch.Tensor):
                # Use tensor statistics for hashing
                context_summary[key] = {
                    'shape': list(value.shape),
                    'mean': float(value.mean().item()) if value.numel() > 0 else 0.0,
                    'std': float(value.std().item()) if value.numel() > 0 else 0.0
                }
            elif isinstance(value, dict):
                context_summary[key] = self._hash_context(value)
            else:
                context_summary[key] = str(value)
        
        return hashlib.md5(json.dumps(context_summary, sort_keys=True).encode()).hexdigest()[:16]
    
    def _compute_dynamic_embedding(self, fragment_data: Dict[str, Any], context_data: Dict[str, Any]) -> torch.Tensor:
        """Compute dynamic embedding using the full CST pipeline"""
        
        # Extract fragment encoding
        fragment_encoding = self.fragment_encoder(
            fragment_data['fragment_chars'],
            fragment_data['context_chars'], 
            fragment_data.get('fragment_positions')
        )
        
        # Fuse with contextual information
        fused_representation = self.information_fuser(fragment_encoding, context_data)
        
        # Project to output space
        output_embedding = self.projection_head(fused_representation)
        
        return output_embedding
    
    def forward(self, 
                text_fragments: torch.Tensor, 
                context_data: Dict[str, Any],
                fragment_chars: Optional[torch.Tensor] = None,
                context_chars: Optional[torch.Tensor] = None,
                fragment_frequencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main forward pass of CST module
        
        Args:
            text_fragments: [batch_size, seq_len] - Token IDs
            context_data: Dictionary of contextual information
            fragment_chars: [batch_size, seq_len, char_len] - Character-level data
            context_chars: [batch_size, seq_len, context_char_len] - Context characters
            fragment_frequencies: [batch_size, seq_len] - Fragment frequencies
        """
        start_time = time.time() if self.enable_profiling else 0
        
        batch_size, seq_len = text_fragments.shape
        device = text_fragments.device
        
        # Initialize output
        output_vectors = torch.zeros(batch_size, seq_len, self.config.d_model, device=device)
        
        for i in range(seq_len):
            fragment_ids = text_fragments[:, i]
            
            # Prepare fragment data
            fragment_data = {
                'fragment_id': fragment_ids,
                'fragment_chars': fragment_chars[:, i] if fragment_chars is not None else None,
                'context_chars': context_chars[:, i] if context_chars is not None else None,
                'fragment_positions': torch.full((batch_size,), i, device=device)
            }
            
            # Prepare context features for ambiguity classification
            context_features = torch.zeros(batch_size, self.config.context_feature_dim, device=device)
            if 'document_embedding' in context_data:
                doc_emb = context_data['document_embedding']
                feature_dim = min(self.config.context_feature_dim, doc_emb.size(-1))
                context_features[:, :feature_dim] = doc_emb[:, :feature_dim]
            
            # Determine if dynamic processing is needed
            freqs = fragment_frequencies[:, i] if fragment_frequencies is not None else None
            is_ambiguous = self.ambiguity_classifier(fragment_ids, context_features, freqs)
            
            # Process each sample in the batch
            for b in range(batch_size):
                if is_ambiguous[b]:
                    # Try cache first
                    sample_fragment_data = {k: v[b] if v is not None else None for k, v in fragment_data.items()}
                    sample_context_data = {k: v[b] if isinstance(v, torch.Tensor) else v for k, v in context_data.items()}
                    
                    cache_key = self._compute_cache_key(sample_fragment_data, sample_context_data)
                    cached_vector = self.cache.get(cache_key)
                    
                    if cached_vector is not None:
                        output_vectors[b, i] = cached_vector
                        if self.enable_profiling:
                            self.profile_stats['cache_hits'] += 1
                    else:
                        # Compute dynamic embedding
                        dynamic_vector = self._compute_dynamic_embedding(sample_fragment_data, sample_context_data)
                        output_vectors[b, i] = dynamic_vector.squeeze(0) if dynamic_vector.dim() > 1 else dynamic_vector
                        
                        # Cache the result
                        self.cache.put(cache_key, output_vectors[b, i])
                        
                        if self.enable_profiling:
                            self.profile_stats['cache_misses'] += 1
                            self.profile_stats['ambiguous_tokens'] += 1
                else:
                    # Use static embedding
                    output_vectors[b, i] = self.static_embeddings(fragment_ids[b])
                    if self.enable_profiling:
                        self.profile_stats['static_tokens'] += 1
        
        if self.enable_profiling:
            self.profile_stats['total_forward_time'] += time.time() - start_time
            self.profile_stats['num_forward_calls'] += 1
        
        return output_vectors
    
    def encode_single_fragment(self, fragment_text: str, context_data: Dict[str, Any]) -> torch.Tensor:
        """Encode a single text fragment (useful for inference)"""
        # This would need proper text preprocessing - simplified for now
        fragment_id = hash(fragment_text) % self.config.vocab_size  # Simplified tokenization
        fragment_tensor = torch.tensor([[fragment_id]], dtype=torch.long)
        
        return self.forward(fragment_tensor, context_data).squeeze()
    
    def enable_profiling_mode(self, enable: bool = True):
        """Enable or disable performance profiling"""
        self.enable_profiling = enable
        if enable:
            # Reset stats
            self.profile_stats = {k: 0 if isinstance(v, (int, float)) else v for k, v in self.profile_stats.items()}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.profile_stats.copy()
        cache_stats = self.cache.stats()
        stats.update(cache_stats)
        
        # Add derived metrics
        if stats['num_forward_calls'] > 0:
            stats['avg_forward_time'] = stats['total_forward_time'] / stats['num_forward_calls']
        
        total_tokens = stats['ambiguous_tokens'] + stats['static_tokens']
        if total_tokens > 0:
            stats['ambiguous_ratio'] = stats['ambiguous_tokens'] / total_tokens
            stats['static_ratio'] = stats['static_tokens'] / total_tokens
        
        return stats
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.cache.clear()
    
    def save_ambiguous_vocab(self, filepath: str):
        """Save the current ambiguous vocabulary"""
        vocab_list = self.ambiguous_vocab.cpu().numpy().tolist()
        with open(filepath, 'w') as f:
            json.dump(vocab_list, f)
    
    def load_ambiguous_vocab(self, filepath: str):
        """Load ambiguous vocabulary from file"""
        with open(filepath, 'r') as f:
            vocab_list = json.load(f)
        self.ambiguity_classifier.update_ambiguous_vocab(vocab_list)


def test_cst_module():
    """Test the complete CST module"""
    from config import CSTConfig
    
    config = CSTConfig()
    config.ambiguous_word_ids = [1, 5, 10, 15, 20]  # Sample ambiguous words
    
    cst = CSTModule(config)
    cst.enable_profiling_mode(True)
    
    batch_size = 2
    seq_len = 8
    
    # Sample input
    text_fragments = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    fragment_chars = torch.randint(0, config.char_vocab_size, (batch_size, seq_len, 32))
    context_chars = torch.randint(0, config.char_vocab_size, (batch_size, seq_len, 64))
    
    context_data = {
        'document_embedding': torch.randn(batch_size, config.raw_doc_dim),
        'metadata': {
            'author': torch.randint(0, config.num_authors, (batch_size,)),
            'domain': torch.randint(0, config.num_domains, (batch_size,)),
        }
    }
    
    # Forward pass
    output = cst(text_fragments, context_data, fragment_chars, context_chars)
    
    print(f"Input shape: {text_fragments.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: {(batch_size, seq_len, config.d_model)}")
    
    # Print performance stats
    stats = cst.get_performance_stats()
    print("\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test caching
    print("\nTesting caching...")
    output2 = cst(text_fragments, context_data, fragment_chars, context_chars)
    
    cache_stats = cst.get_performance_stats()
    print(f"Cache hit rate after second pass: {cache_stats['hit_rate']:.2%}")
    
    assert output.shape == (batch_size, seq_len, config.d_model), \
        f"Expected {(batch_size, seq_len, config.d_model)}, got {output.shape}"
    
    print("CST Module test passed!")


if __name__ == "__main__":
    test_cst_module()