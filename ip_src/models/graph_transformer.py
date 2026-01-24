"""
Heterogeneous Graph Transformer for Instant Policy.

Implements the three-stage network architecture from Appendix C of the paper:
- σ(·): Local Graph Encoder - encodes individual timestep observations
- φ(·): Context Aggregator - aggregates demonstrations with current observation
- ψ(·): Action Decoder - Graph Transformer that outputs geometric flow vectors

Key features following the paper:
1. Edge features in attention: Attn(Q, K+E) · (V+E) as per Equation 3
2. ActionDecoder is a GRAPH Transformer, not sequence Transformer
3. Output is geometric flow vectors [∇p_trans, ∇p_rot], not flat noise

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math


class EdgeAwareAttention(MessagePassing):
    """
    Edge-aware multi-head attention following Equation 3 in the paper.
    
    Attention formula: Attn(Q, K + W_edge · E) · (V + W_edge · E)
    
    This is different from standard Transformer attention as edge features
    are added to both keys and values.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.k_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.v_proj = nn.Linear(in_channels, out_channels, bias=bias)
        
        # Edge feature projection (W_5 in Equation 3)
        # Projects edge features to be added to K and V
        self.edge_proj = nn.Linear(edge_dim, out_channels, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(out_channels, out_channels, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.edge_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with edge-aware attention.
        
        Args:
            x: Node features [N, in_channels] or tuple (src_x, dst_x)
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]
        
        Returns:
            Updated node features [N, out_channels]
        """
        # Handle bipartite case
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
        
        # Project to Q, K, V
        q = self.q_proj(x_dst)  # [N_dst, out_channels]
        k = self.k_proj(x_src)  # [N_src, out_channels]
        v = self.v_proj(x_src)  # [N_src, out_channels]
        
        # Project edge features
        edge_feat = self.edge_proj(edge_attr)  # [E, out_channels]
        
        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        edge_feat = edge_feat.view(-1, self.num_heads, self.head_dim)
        
        # Propagate messages
        out = self.propagate(
            edge_index, 
            q=q, k=k, v=v, 
            edge_feat=edge_feat,
            size=(x_src.size(0), x_dst.size(0))
        )
        
        # Reshape and project output
        out = out.view(-1, self.out_channels)
        out = self.out_proj(out)
        
        return out
    
    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_feat: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """
        Compute messages with edge-aware attention.
        
        Following Equation 3: Attn(Q, K + E) · (V + E)
        """
        # Add edge features to keys and values
        k_with_edge = k_j + edge_feat  # K + W_5 · E
        v_with_edge = v_j + edge_feat  # V + W_5 · E
        
        # Compute attention scores
        attn = (q_i * k_with_edge).sum(dim=-1) * self.scale  # [E, num_heads]
        
        # Softmax over source nodes
        attn = softmax(attn, index, ptr, size_i)  # [E, num_heads]
        attn = self.attn_dropout(attn)
        
        # Weighted sum of values
        out = attn.unsqueeze(-1) * v_with_edge  # [E, num_heads, head_dim]
        
        return out


class HeteroEdgeAwareAttention(nn.Module):
    """
    Heterogeneous edge-aware attention for multiple node/edge types.
    
    Uses separate parameters for each edge type to enable type-specific learning.
    Paper explicitly states homogeneous graphs perform worse.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        node_types: List[str] = None,
        edge_types: List[Tuple[str, str, str]] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        if node_types is None:
            node_types = ["point_cloud", "gripper", "ghost"]
        if edge_types is None:
            edge_types = []
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        # Create attention layer for each edge type
        self.attentions = nn.ModuleDict()
        for edge_type in edge_types:
            edge_name = "__".join(edge_type)
            self.attentions[edge_name] = EdgeAwareAttention(
                in_channels=in_channels,
                out_channels=out_channels,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # Node type projections for nodes without incoming edges
        self.node_projections = nn.ModuleDict({
            node_type: nn.Linear(in_channels, out_channels)
            for node_type in node_types
        })
        
        # Output layer norm
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in node_types
        })
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through heterogeneous attention.
        
        Args:
            x_dict: Node features {node_type: [N, in_channels]}
            edge_index_dict: Edge indices {edge_type: [2, E]}
            edge_attr_dict: Edge features {edge_type: [E, edge_dim]}
        
        Returns:
            Updated node features dictionary
        """
        out_dict = {}
        
        # Initialize with projected input (for nodes without incoming edges)
        for node_type, x in x_dict.items():
            if node_type in self.node_projections:
                out_dict[node_type] = self.node_projections[node_type](x)
        
        # Apply edge-aware attention for each edge type
        for edge_type, edge_index in edge_index_dict.items():
            edge_name = "__".join(edge_type)
            if edge_name not in self.attentions:
                continue
            
            src_type, _, dst_type = edge_type
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            
            # Get edge attributes
            edge_attr = edge_attr_dict.get(edge_type)
            if edge_attr is None:
                continue
            
            # Apply attention
            attn = self.attentions[edge_name]
            out = attn((x_dict[src_type], x_dict[dst_type]), edge_index, edge_attr)
            
            # Accumulate to destination nodes
            if dst_type in out_dict:
                out_dict[dst_type] = out_dict[dst_type] + out
            else:
                out_dict[dst_type] = out
        
        # Apply layer norm
        for node_type in out_dict:
            if node_type in self.norms:
                out_dict[node_type] = self.norms[node_type](out_dict[node_type])
        
        return out_dict


class GraphTransformerLayer(nn.Module):
    """
    A single layer of the Graph Transformer with edge features.
    
    Includes:
    - Edge-aware multi-head attention
    - Feed-forward network
    - Residual connections and layer normalization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        node_types: List[str] = None,
        edge_types: List[Tuple[str, str, str]] = None,
    ):
        super().__init__()
        
        if node_types is None:
            node_types = ["point_cloud", "gripper", "ghost"]
        
        self.node_types = node_types
        
        # Pre-norm layer norms
        self.norm1 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
        self.norm2 = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim)
            for node_type in node_types
        })
        
        # Edge-aware attention
        self.attention = HeteroEdgeAwareAttention(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
            node_types=node_types,
            edge_types=edge_types,
        )
        
        # Feed-forward networks for each node type
        self.ffn = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * ff_mult, hidden_dim),
                nn.Dropout(dropout),
            )
            for node_type in node_types
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through transformer layer."""
        
        # Pre-norm attention
        normed_x = {
            k: self.norm1[k](v) if k in self.norm1 else v 
            for k, v in x_dict.items()
        }
        attn_out = self.attention(normed_x, edge_index_dict, edge_attr_dict)
        
        # Residual connection
        x_dict = {
            k: x_dict[k] + self.dropout(attn_out.get(k, torch.zeros_like(x_dict[k])))
            for k in x_dict
        }
        
        # Pre-norm FFN
        out_dict = {}
        for k, v in x_dict.items():
            if k in self.ffn:
                normed = self.norm2[k](v)
                out_dict[k] = v + self.ffn[k](normed)
            else:
                out_dict[k] = v
        
        return out_dict


class GraphTransformer(nn.Module):
    """
    Multi-layer Graph Transformer with edge features.
    
    Used for all three stages (σ, φ, ψ) with different configurations.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        ff_mult: int = 4,
        node_types: List[str] = None,
        edge_types: List[Tuple[str, str, str]] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_mult=ff_mult,
                node_types=node_types,
                edge_types=edge_types,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all transformer layers."""
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        return x_dict


class LocalGraphEncoder(nn.Module):
    """
    σ(·): Local Graph Encoder
    
    Encodes a single timestep observation (point cloud + gripper state)
    into node embeddings using edge-aware graph transformer.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Node types for local graph
        node_types = ["point_cloud", "gripper"]
        
        # Edge types for local graph
        edge_types = [
            ("point_cloud", "pc_to_pc", "point_cloud"),
            ("point_cloud", "pc_to_gripper", "gripper"),
            ("gripper", "gripper_to_pc", "point_cloud"),
            ("gripper", "gripper_same", "gripper"),
        ]
        
        self.transformer = GraphTransformer(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            node_types=node_types,
            edge_types=edge_types,
        )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Encode local graph."""
        return self.transformer(x_dict, edge_index_dict, edge_attr_dict)


class ContextAggregator(nn.Module):
    """
    φ(·): Context Aggregator
    
    Aggregates encoded demonstrations with the current observation.
    Uses temporal edges and demo-to-live edges (NOT all-to-all).
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Node types
        node_types = ["point_cloud", "gripper"]
        
        # Edge types including temporal and demo-to-live
        edge_types = [
            ("point_cloud", "pc_to_pc", "point_cloud"),
            ("point_cloud", "pc_to_gripper", "gripper"),
            ("gripper", "gripper_to_pc", "point_cloud"),
            ("gripper", "gripper_same", "gripper"),
            ("gripper", "temporal", "gripper"),  # t -> t+1 within demo
            ("gripper", "demo_to_live", "gripper"),  # Demo to live
        ]
        
        self.transformer = GraphTransformer(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            node_types=node_types,
            edge_types=edge_types,
        )
        
        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate context."""
        out_dict = self.transformer(x_dict, edge_index_dict, edge_attr_dict)
        
        # Apply bottleneck to gripper features
        if "gripper" in out_dict:
            out_dict["gripper"] = self.bottleneck(out_dict["gripper"])
        
        return out_dict


class ActionDecoder(nn.Module):
    """
    ψ(·): Action Decoder - Graph Transformer
    
    This is a GRAPH TRANSFORMER, not a sequence transformer!
    
    - Input: Complete HeteroData graph with connected ghost gripper nodes
    - Process: Message passing to update ghost node features
    - Output: Geometric flow vectors [∇p_trans, ∇p_rot] for each ghost node
    
    Output is 6D per ghost node: 3D translation flow + 3D rotation flow
    (not 9D flat noise)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        flow_dim: int = 6,  # 3D translation + 3D rotation flow
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.flow_dim = flow_dim
        
        # Node types including ghost nodes
        node_types = ["point_cloud", "gripper", "ghost"]
        
        # Edge types including ghost connections
        edge_types = [
            ("point_cloud", "pc_to_pc", "point_cloud"),
            ("point_cloud", "pc_to_gripper", "gripper"),
            ("gripper", "gripper_to_pc", "point_cloud"),
            ("gripper", "gripper_same", "gripper"),
            ("gripper", "temporal", "gripper"),
            ("gripper", "demo_to_live", "gripper"),
            ("gripper", "live_to_ghost", "ghost"),  # Live -> Ghost
            ("ghost", "ghost_to_live", "gripper"),  # Ghost -> Live
            ("ghost", "ghost_to_ghost", "ghost"),  # Between ghosts
        ]
        
        self.transformer = GraphTransformer(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            node_types=node_types,
            edge_types=edge_types,
        )
        
        # Output head for geometric flow prediction
        # Predicts flow vectors for ghost nodes
        self.flow_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flow_dim),  # 3D trans flow + 3D rot flow
        )
        
        # Gripper state prediction head (binary open/close)
        self.gripper_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict geometric flow vectors for ghost nodes.
        
        Args:
            x_dict: Node features including ghost nodes
            edge_index_dict: All edges including ghost connections
            edge_attr_dict: Edge features
        
        Returns:
            flow: Geometric flow vectors [num_ghost, 6] (3D trans + 3D rot)
            gripper: Gripper state predictions [num_ghost, 1]
        """
        # Run graph transformer
        out_dict = self.transformer(x_dict, edge_index_dict, edge_attr_dict)
        
        # Extract ghost node features
        ghost_features = out_dict["ghost"]  # [num_ghost, hidden_dim]
        
        # Predict flow vectors
        flow = self.flow_head(ghost_features)  # [num_ghost, 6]
        
        # Predict gripper states
        gripper = self.gripper_head(ghost_features)  # [num_ghost, 1]
        
        return flow, gripper
