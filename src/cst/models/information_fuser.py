"""
Information Fuser for CST
Fuses fragment encodings with multimodal and document-level signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math


class DocumentEncoder(nn.Module):
    """Lightweight encoder for document-level features"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)


class MetadataProcessor(nn.Module):
    """Process various types of metadata"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Categorical metadata embeddings
        self.categorical_processors = nn.ModuleDict({
            'author': nn.Embedding(config.num_authors, config.embed_dim),
            'domain': nn.Embedding(config.num_domains, config.embed_dim),
            'language': nn.Embedding(100, config.embed_dim),  # Support 100 languages
            'genre': nn.Embedding(50, config.embed_dim),      # 50 genres
        })
        
        # Continuous metadata processors
        self.continuous_processors = nn.ModuleDict({
            'timestamp': nn.Sequential(
                nn.Linear(1, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, config.embed_dim)
            ),
            'document_length': nn.Sequential(
                nn.Linear(1, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, config.embed_dim)
            ),
            'readability_score': nn.Sequential(
                nn.Linear(1, config.embed_dim // 2),
                nn.GELU(),
                nn.Linear(config.embed_dim // 2, config.embed_dim)
            )
        })
        
        # Text-based metadata processor (for titles, descriptions, etc.)
        self.text_processor = nn.Sequential(
            nn.Linear(config.d_model, config.embed_dim),
            nn.GELU()
        )
        
    def forward(self, metadata: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Process metadata and return list of feature tensors"""
        features = []
        
        # Process categorical features
        for key, processor in self.categorical_processors.items():
            if key in metadata:
                feat = processor(metadata[key])
                features.append(feat)
        
        # Process continuous features
        for key, processor in self.continuous_processors.items():
            if key in metadata:
                # Normalize continuous values
                value = metadata[key]
                if key == 'timestamp':
                    # Normalize timestamp to reasonable range
                    value = (value - 1577836800) / (365.25 * 24 * 3600)  # Years since 2020
                elif key == 'document_length':
                    value = torch.log(value.float() + 1)  # Log normalization
                elif key == 'readability_score':
                    value = value / 100.0  # Assume 0-100 scale
                
                feat = processor(value.unsqueeze(-1) if value.dim() == 1 else value)
                features.append(feat)
        
        # Process text-based metadata
        if 'title_embedding' in metadata:
            feat = self.text_processor(metadata['title_embedding'])
            features.append(feat)
            
        if 'description_embedding' in metadata:
            feat = self.text_processor(metadata['description_embedding'])
            features.append(feat)
        
        return features


class MultimodalProcessor(nn.Module):
    """Process multimodal signals (images, audio, etc.)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Image processors for different encodings
        self.image_processors = nn.ModuleDict({
            'clip': nn.Sequential(
                nn.Linear(config.clip_dim, config.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_dim, config.embed_dim)
            ),
            'resnet': nn.Sequential(
                nn.Linear(2048, config.hidden_dim),  # ResNet-50 features
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.embed_dim)
            ),
            'vit': nn.Sequential(
                nn.Linear(768, config.hidden_dim),   # ViT features
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.embed_dim)
            )
        })
        
        # Audio processors
        self.audio_processors = nn.ModuleDict({
            'wav2vec': nn.Sequential(
                nn.Linear(config.audio_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.embed_dim)
            ),
            'mel_spectrogram': nn.Sequential(
                nn.Linear(128, config.hidden_dim),   # Mel features
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.embed_dim)
            )
        })
        
        # Video processor (if available)
        self.video_processor = nn.Sequential(
            nn.Linear(1024, config.hidden_dim),  # Video features
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embed_dim)
        )
        
    def forward(self, multimodal_data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Process multimodal data and return feature tensors"""
        features = []
        
        # Process images
        for img_type, processor in self.image_processors.items():
            key = f'image_{img_type}'
            if key in multimodal_data:
                feat = processor(multimodal_data[key])
                features.append(feat)
        
        # Process audio
        for audio_type, processor in self.audio_processors.items():
            key = f'audio_{audio_type}'
            if key in multimodal_data:
                feat = processor(multimodal_data[key])
                features.append(feat)
        
        # Process video
        if 'video_features' in multimodal_data:
            feat = self.video_processor(multimodal_data['video_features'])
            features.append(feat)
        
        return features


class CrossModalAttention(nn.Module):
    """Cross-attention mechanism for fusing different modalities"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, query, key_value):
        """
        Args:
            query: [batch_size, 1, d_model] - Fragment representation
            key_value: [batch_size, n_context, d_model] - Context representations
        """
        # Cross attention
        attn_output, attn_weights = self.multihead_attn(query, key_value, key_value)
        query = self.norm1(query + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)
        
        return query, attn_weights


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns importance weights for different modalities"""
    
    def __init__(self, input_dims: List[int], output_dim: int, num_modalities: int):
        super().__init__()
        
        # Project all inputs to same dimension
        self.projectors = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        
        # Attention mechanism for modality weighting
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, 1)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Adaptively fuse multiple feature vectors"""
        if not features:
            raise ValueError("No features provided for fusion")
        
        # Project to common dimension
        projected = [proj(feat) for proj, feat in zip(self.projectors, features)]
        
        if len(projected) == 1:
            return self.fusion(projected[0])
        
        # Stack for attention computation
        stacked = torch.stack(projected, dim=1)  # [batch, n_modalities, dim]
        
        # Compute attention weights
        attn_scores = self.attention(stacked)  # [batch, n_modalities, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted fusion
        fused = (stacked * attn_weights).sum(dim=1)
        
        return self.fusion(fused)


class InformationFuser(nn.Module):
    """
    Main Information Fuser module that combines fragment encodings with 
    document-level, metadata, and multimodal signals
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Component processors
        self.doc_encoder = DocumentEncoder(
            input_dim=config.raw_doc_dim,
            output_dim=config.document_encoding_dim,
            hidden_dim=config.hidden_dim
        )
        
        self.metadata_processor = MetadataProcessor(config)
        self.multimodal_processor = MultimodalProcessor(config)
        
        # Cross-attention for fragment-context interaction
        self.cross_attention = CrossModalAttention(
            d_model=config.fragment_encoding_dim,
            num_heads=8
        )
        
        # Determine fusion dimensions dynamically based on available features
        max_features = (
            1 +  # Fragment encoding
            1 +  # Document encoding  
            len(self.metadata_processor.categorical_processors) +
            len(self.metadata_processor.continuous_processors) +
            2 +  # Text metadata (title, description)
            3 +  # Image processors
            2 +  # Audio processors
            1    # Video processor
        )
        
        # Adaptive fusion
        self.adaptive_fusion = AdaptiveFusion(
            input_dims=[config.fragment_encoding_dim] * max_features,
            output_dim=config.fused_dim,
            num_modalities=max_features
        )
        
        # Alternative: Simple concatenation fusion
        self.concat_fusion = nn.Sequential(
            nn.Linear(config.fragment_encoding_dim * max_features, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.fused_dim)
        )
        
        # Gating mechanism to choose fusion strategy
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.fragment_encoding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, fragment_encoding: torch.Tensor, context_data: Dict[str, Any]) -> torch.Tensor:
        """
        Fuse fragment encoding with contextual information
        
        Args:
            fragment_encoding: [batch_size, fragment_encoding_dim]
            context_data: Dictionary containing various context signals
        """
        batch_size = fragment_encoding.size(0)
        fusion_features = [fragment_encoding]
        
        # Process document-level signals
        if 'document_embedding' in context_data:
            doc_features = self.doc_encoder(context_data['document_embedding'])
            fusion_features.append(doc_features)
        
        # Process metadata
        if 'metadata' in context_data:
            meta_features = self.metadata_processor(context_data['metadata'])
            fusion_features.extend(meta_features)
        
        # Process multimodal signals
        if 'multimodal' in context_data:
            mm_features = self.multimodal_processor(context_data['multimodal'])
            fusion_features.extend(mm_features)
        
        # Cross-attention enhancement if we have context
        if len(fusion_features) > 1:
            context_features = torch.stack(fusion_features[1:], dim=1)  # [batch, n_context, dim]
            fragment_query = fragment_encoding.unsqueeze(1)  # [batch, 1, dim]
            
            # Pad or project features to match fragment_encoding dimension
            if context_features.size(-1) != fragment_encoding.size(-1):
                context_proj = nn.Linear(context_features.size(-1), fragment_encoding.size(-1))
                if hasattr(self, '_context_proj'):
                    context_proj = self._context_proj
                else:
                    self._context_proj = context_proj.to(fragment_encoding.device)
                    context_proj = self._context_proj
                context_features = context_proj(context_features)
            
            enhanced_fragment, attn_weights = self.cross_attention(fragment_query, context_features)
            fusion_features[0] = enhanced_fragment.squeeze(1)
        
        # Adaptive vs concatenation fusion based on gate
        gate_score = self.fusion_gate(fragment_encoding)
        
        # Pad features to maximum expected size for consistent processing
        max_expected = 12  # Reasonable maximum based on our processors
        while len(fusion_features) < max_expected:
            # Add zero features for missing modalities
            zero_feat = torch.zeros_like(fusion_features[0])
            fusion_features.append(zero_feat)
        
        # Truncate if somehow we have too many
        fusion_features = fusion_features[:max_expected]
        
        # Adaptive fusion
        try:
            adaptive_output = self.adaptive_fusion(fusion_features)
        except Exception:
            # Fallback to simple mean if adaptive fusion fails
            adaptive_output = torch.stack(fusion_features).mean(dim=0)
        
        # Concatenation fusion
        concat_input = torch.cat(fusion_features, dim=-1)
        concat_output = self.concat_fusion(concat_input)
        
        # Gated combination
        final_output = gate_score * adaptive_output + (1 - gate_score) * concat_output
        
        return final_output
    
    def get_attention_weights(self):
        """Return the last computed attention weights for interpretability"""
        if hasattr(self.cross_attention, 'last_attn_weights'):
            return self.cross_attention.last_attn_weights
        return None


class ContextExtractor:
    """Utility class to extract relevant context from various data sources"""
    
    @staticmethod
    def extract_document_context(text: str, fragment_position: int, window_size: int = 256) -> Dict[str, Any]:
        """Extract document-level context around a fragment"""
        start = max(0, fragment_position - window_size)
        end = min(len(text), fragment_position + window_size)
        
        return {
            'surrounding_text': text[start:end],
            'document_stats': {
                'total_length': len(text),
                'fragment_relative_position': fragment_position / len(text),
                'paragraph_number': text[:fragment_position].count('\n\n') + 1
            }
        }
    
    @staticmethod
    def extract_metadata_context(document_metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert document metadata to tensor format"""
        tensor_metadata = {}
        
        # Handle different metadata types
        for key, value in document_metadata.items():
            if isinstance(value, (int, float)):
                tensor_metadata[key] = torch.tensor([value], dtype=torch.float32)
            elif isinstance(value, str):
                # For string values, you might want to hash or use a lookup table
                # For now, we'll skip them or handle specific cases
                if key in ['author', 'domain', 'language', 'genre']:
                    # These should be converted to IDs by the calling code
                    if isinstance(value, int):
                        tensor_metadata[key] = torch.tensor([value], dtype=torch.long)
            elif isinstance(value, torch.Tensor):
                tensor_metadata[key] = value
        
        return tensor_metadata
    
    @staticmethod
    def prepare_context_data(
        fragment_text: str,
        document_text: str,
        fragment_position: int,
        metadata: Optional[Dict[str, Any]] = None,
        multimodal_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """Prepare complete context data for information fusion"""
        
        context_data = {}
        
        # Document context
        doc_context = ContextExtractor.extract_document_context(
            document_text, fragment_position
        )
        
        # Create document embedding (simplified - in practice, use proper encoder)
        doc_text = doc_context['surrounding_text']
        # This would typically be computed by a separate document encoder
        context_data['document_embedding'] = torch.randn(1024)  # Placeholder
        
        # Metadata context
        if metadata:
            context_data['metadata'] = ContextExtractor.extract_metadata_context(metadata)
        
        # Multimodal context
        if multimodal_data:
            context_data['multimodal'] = multimodal_data
        
        return context_data


# Example usage and testing functions
def test_information_fuser():
    """Test the InformationFuser with sample data"""
    from config import CSTConfig
    
    config = CSTConfig()
    fuser = InformationFuser(config)
    
    batch_size = 4
    
    # Sample fragment encoding
    fragment_encoding = torch.randn(batch_size, config.fragment_encoding_dim)
    
    # Sample context data
    context_data = {
        'document_embedding': torch.randn(batch_size, config.raw_doc_dim),
        'metadata': {
            'author': torch.randint(0, config.num_authors, (batch_size,)),
            'domain': torch.randint(0, config.num_domains, (batch_size,)),
            'timestamp': torch.randn(batch_size),
            'document_length': torch.randint(100, 10000, (batch_size,)).float(),
        },
        'multimodal': {
            'image_clip': torch.randn(batch_size, config.clip_dim),
            'audio_wav2vec': torch.randn(batch_size, config.audio_dim),
        }
    }
    
    # Forward pass
    fused_output = fuser(fragment_encoding, context_data)
    
    print(f"Input fragment encoding shape: {fragment_encoding.shape}")
    print(f"Output fused representation shape: {fused_output.shape}")
    print(f"Expected output dimension: {config.fused_dim}")
    
    assert fused_output.shape == (batch_size, config.fused_dim), \
        f"Expected {(batch_size, config.fused_dim)}, got {fused_output.shape}"
    
    print("Information Fuser test passed!")


if __name__ == "__main__":
    test_information_fuser()