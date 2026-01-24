"""
Graph Builder for Instant Policy.

Implements the graph representation described in Section 3.2 of the paper:
- Local Graph: Represents a single observation (point cloud + end-effector state)
- Context Graph: Aggregates multiple demonstrations with the current observation
- Ghost Gripper Nodes: Action hypotheses as future keypoint nodes

Key features:
- NeRF-like sinusoidal positional encoding (Appendix A)
- Edge features with relative position encoding
- Temporal edges connecting consecutive gripper nodes within demonstrations
- Ghost gripper nodes for action representation

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from torch_geometric.data import HeteroData
from torch_geometric.nn import knn_graph
import math


@dataclass
class LocalGraph:
    """
    Local Graph representation for a single timestep observation.
    
    Contains:
    - Point cloud nodes with geometric features
    - Gripper nodes (6 nodes representing end-effector pose)
    - Gripper state embedded in node features
    """
    data: HeteroData
    num_points: int
    num_gripper_nodes: int = 6  # 6 nodes to represent gripper pose


@dataclass
class ContextGraph:
    """
    Context Graph that aggregates demonstrations and current observation.
    
    Contains:
    - Multiple Local Graphs from demonstrations with TEMPORAL edges
    - Current observation Local Graph
    - Cross-graph edges (demo-to-live only, NOT all-to-all)
    - Ghost gripper nodes for action prediction
    """
    data: HeteroData
    num_demos: int
    num_waypoints_per_demo: int
    num_ghost_nodes: int  # Number of future keypoint predictions


def sinusoidal_positional_encoding(
    coords: torch.Tensor,
    num_frequencies: int = 10,
    include_input: bool = True,
) -> torch.Tensor:
    """
    NeRF-like sinusoidal positional encoding as described in Appendix A.
    
    Encodes position p as: [sin(2^0 π p), cos(2^0 π p), sin(2^1 π p), cos(2^1 π p), ...]
    
    Args:
        coords: Input coordinates [..., D] (e.g., 3D positions)
        num_frequencies: Number of frequency bands (L in the paper)
        include_input: Whether to include original coordinates
    
    Returns:
        Encoded coordinates [..., D * (2 * num_frequencies + include_input)]
    """
    # Frequency bands: 2^0, 2^1, ..., 2^(L-1)
    freq_bands = 2.0 ** torch.arange(num_frequencies, device=coords.device, dtype=coords.dtype)
    
    # Scale coordinates by pi
    scaled_coords = coords.unsqueeze(-1) * freq_bands * math.pi  # [..., D, L]
    
    # Apply sin and cos
    sin_enc = torch.sin(scaled_coords)  # [..., D, L]
    cos_enc = torch.cos(scaled_coords)  # [..., D, L]
    
    # Interleave sin and cos: [sin(f0), cos(f0), sin(f1), cos(f1), ...]
    encoding = torch.stack([sin_enc, cos_enc], dim=-1)  # [..., D, L, 2]
    encoding = encoding.reshape(*coords.shape[:-1], -1)  # [..., D * L * 2]
    
    if include_input:
        encoding = torch.cat([coords, encoding], dim=-1)
    
    return encoding


def compute_edge_features(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_frequencies: int = 6,
) -> torch.Tensor:
    """
    Compute edge features as sinusoidal encoding of relative positions.
    
    Args:
        src_pos: Source node positions [N_src, 3]
        dst_pos: Destination node positions [N_dst, 3]
        edge_index: Edge indices [2, E]
        num_frequencies: Number of frequency bands for encoding
    
    Returns:
        Edge features [E, feature_dim]
    """
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    
    # Compute relative position: Δp = p_dst - p_src
    delta_p = dst_pos[dst_idx] - src_pos[src_idx]  # [E, 3]
    
    # Also compute distance
    distance = torch.norm(delta_p, dim=-1, keepdim=True)  # [E, 1]
    
    # Apply sinusoidal encoding to relative position
    edge_features = sinusoidal_positional_encoding(
        delta_p, 
        num_frequencies=num_frequencies,
        include_input=True,
    )
    
    # Concatenate distance
    edge_features = torch.cat([edge_features, distance], dim=-1)
    
    return edge_features


class GraphBuilder(nn.Module):
    """
    Builds graph representations for Instant Policy following the paper exactly.
    
    Key differences from standard implementations:
    1. Sinusoidal positional encoding (NeRF-like) instead of linear
    2. Edge features with relative position encoding
    3. Temporal edges within demonstrations (t -> t+1)
    4. Ghost gripper nodes for action representation (NOT flat action tokens)
    5. NO all-to-all demo connections - only within-demo temporal + demo-to-live
    """
    
    # Node types
    NODE_TYPE_POINT_CLOUD = "point_cloud"
    NODE_TYPE_GRIPPER = "gripper"  # End-effector/gripper nodes (6 per timestep)
    NODE_TYPE_GHOST = "ghost"  # Ghost gripper nodes for action prediction
    
    # Edge types (following paper Figure 2)
    EDGE_TYPE_PC_PC = ("point_cloud", "pc_to_pc", "point_cloud")  # k-NN within point cloud
    EDGE_TYPE_PC_GRIPPER = ("point_cloud", "pc_to_gripper", "gripper")  # Point cloud to gripper
    EDGE_TYPE_GRIPPER_PC = ("gripper", "gripper_to_pc", "point_cloud")  # Gripper to point cloud
    EDGE_TYPE_GRIPPER_GRIPPER_SAME = ("gripper", "gripper_same", "gripper")  # Within same timestep
    EDGE_TYPE_TEMPORAL = ("gripper", "temporal", "gripper")  # t -> t+1 within demo (RED lines in paper)
    EDGE_TYPE_DEMO_LIVE = ("gripper", "demo_to_live", "gripper")  # Demo gripper to live gripper
    EDGE_TYPE_LIVE_GHOST = ("gripper", "live_to_ghost", "ghost")  # Live gripper to ghost
    EDGE_TYPE_GHOST_LIVE = ("ghost", "ghost_to_live", "gripper")  # Ghost to live gripper
    EDGE_TYPE_GHOST_GHOST = ("ghost", "ghost_to_ghost", "ghost")  # Between ghost nodes
    
    def __init__(
        self,
        num_points: int = 16,
        k_neighbors: int = 8,
        num_gripper_nodes: int = 6,
        feature_dim: int = 256,
        num_pos_frequencies: int = 10,
        num_edge_frequencies: int = 6,
        prediction_horizon: int = 8,
        device: str = "cuda",
    ):
        """
        Initialize the Graph Builder.
        
        Args:
            num_points: Number of keypoint nodes M from geometry encoder (default: 16)
                       This is the number of FPS-sampled centroids, NOT raw point cloud size
            k_neighbors: Number of neighbors for k-NN graph construction (default: 8)
                        Should be < num_points to avoid self-loops
            num_gripper_nodes: Number of gripper nodes per timestep (default: 6)
            feature_dim: Dimension of node features
            num_pos_frequencies: Frequency bands for position encoding
            num_edge_frequencies: Frequency bands for edge encoding
            prediction_horizon: Number of future keypoints to predict
            device: Device to build graphs on
        """
        super().__init__()
        self.num_points = num_points
        self.k_neighbors = k_neighbors
        self.num_gripper_nodes = num_gripper_nodes
        self.feature_dim = feature_dim
        self.num_pos_frequencies = num_pos_frequencies
        self.num_edge_frequencies = num_edge_frequencies
        self.prediction_horizon = prediction_horizon
        self.device = device
        
        # Compute encoding dimensions
        # Position encoding: 3 * (2 * L + 1) for 3D with L frequencies + original
        self.pos_encoding_dim = 3 * (2 * num_pos_frequencies + 1)
        # Edge encoding: 3 * (2 * L + 1) + 1 (distance)
        self.edge_encoding_dim = 3 * (2 * num_edge_frequencies + 1) + 1
        
        # Learnable gripper node embeddings (6 nodes with different roles)
        # Following paper: 2 position tokens + 4 orientation tokens
        self.gripper_type_embedding = nn.Embedding(num_gripper_nodes, feature_dim)
        
        # Gripper state embedding (open/closed)
        self.gripper_state_embedding = nn.Embedding(2, feature_dim)
        
        # Ghost node type embedding (for action prediction)
        self.ghost_type_embedding = nn.Embedding(prediction_horizon, feature_dim)
        
        # Project position encoding to feature dimension
        self.pos_projection = nn.Linear(self.pos_encoding_dim, feature_dim)
        
        # Project edge features to feature dimension
        self.edge_projection = nn.Linear(self.edge_encoding_dim, feature_dim)
        
        # Timestep embedding for diffusion (sinusoidal)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.SiLU(),
            nn.Linear(feature_dim * 4, feature_dim),
        )
    
    def _get_sinusoidal_timestep_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """Get sinusoidal embedding for diffusion timestep."""
        half_dim = self.feature_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.timestep_mlp(emb)
    
    def build_local_graph(
        self,
        point_cloud: torch.Tensor,
        point_features: torch.Tensor,
        gripper_pose: torch.Tensor,
        gripper_state: torch.Tensor,
    ) -> LocalGraph:
        """
        Build a local graph for a single observation.
        
        Args:
            point_cloud: Keypoint positions [M, 3] from geometry encoder (M=16 typically)
                        These are the sampled centroids, NOT the raw 2048-point cloud
            point_features: Keypoint features [M, feature_dim] from geometry encoder
            gripper_pose: End-effector pose [4, 4] transformation matrix
            gripper_state: Gripper state [1] (0: closed, 1: open)
        
        Returns:
            LocalGraph containing the hetero graph data
            
        Note:
            The point_cloud and point_features come from PointNetPlusPlusEncoder,
            which samples M=16 keypoints using FPS and encodes local geometry
            around each keypoint with NeRF-like positional encoding (Appendix A).
        """
        data = HeteroData()
        device = point_cloud.device
        
        # === Point Cloud Nodes ===
        data[self.NODE_TYPE_POINT_CLOUD].x = point_features
        data[self.NODE_TYPE_POINT_CLOUD].pos = point_cloud
        
        # === Gripper Nodes ===
        # Extract position and rotation from pose matrix
        gripper_position = gripper_pose[:3, 3]  # [3]
        gripper_rotation = gripper_pose[:3, :3]  # [3, 3]
        
        # Create 6 gripper node positions based on the gripper frame
        # Following paper: nodes represent gripper structure
        gripper_positions = self._create_gripper_node_positions(
            gripper_position, gripper_rotation
        )  # [6, 3]
        
        # Create gripper node features
        # 1. Type embedding (which of the 6 nodes)
        node_types = torch.arange(self.num_gripper_nodes, device=device)
        type_emb = self.gripper_type_embedding(node_types)  # [6, feature_dim]
        
        # 2. Position encoding (sinusoidal)
        pos_enc = sinusoidal_positional_encoding(
            gripper_positions, 
            num_frequencies=self.num_pos_frequencies
        )
        pos_features = self.pos_projection(pos_enc)  # [6, feature_dim]
        
        # 3. Gripper state embedding
        # Ensure gripper_state is a scalar or 0-dim tensor for embedding lookup
        if gripper_state.dim() > 0:
            gripper_state = gripper_state.squeeze()  # Remove extra dimensions
        state_emb = self.gripper_state_embedding(gripper_state.long())  # [feature_dim]
        
        # Combine features (broadcast state_emb to all gripper nodes)
        gripper_features = type_emb + pos_features + state_emb.unsqueeze(0)  # [6, feature_dim]
        
        data[self.NODE_TYPE_GRIPPER].x = gripper_features
        data[self.NODE_TYPE_GRIPPER].pos = gripper_positions
        
        # === Edges with Features ===
        
        # 1. Point cloud k-NN edges
        # Dynamically adjust k to ensure k < num_points (avoid self-loops)
        num_pc = point_cloud.shape[0]
        k_actual = min(self.k_neighbors, num_pc - 1)
        pc_edges = knn_graph(point_cloud, k=k_actual, loop=False)
        data[self.EDGE_TYPE_PC_PC].edge_index = pc_edges
        # Compute edge features
        pc_edge_features = compute_edge_features(
            point_cloud, point_cloud, pc_edges,
            num_frequencies=self.num_edge_frequencies
        )
        data[self.EDGE_TYPE_PC_PC].edge_attr = self.edge_projection(pc_edge_features)
        
        # 2. Point cloud to gripper edges (all keypoints connect to all gripper nodes)
        pc_to_gripper_src = torch.arange(num_pc, device=device).repeat_interleave(self.num_gripper_nodes)
        pc_to_gripper_dst = torch.arange(self.num_gripper_nodes, device=device).repeat(num_pc)
        pc_to_gripper_edges = torch.stack([pc_to_gripper_src, pc_to_gripper_dst])
        data[self.EDGE_TYPE_PC_GRIPPER].edge_index = pc_to_gripper_edges
        # Edge features
        pc_gripper_edge_features = compute_edge_features(
            point_cloud, gripper_positions, pc_to_gripper_edges,
            num_frequencies=self.num_edge_frequencies
        )
        data[self.EDGE_TYPE_PC_GRIPPER].edge_attr = self.edge_projection(pc_gripper_edge_features)
        
        # 3. Gripper to point cloud edges (reverse)
        gripper_to_pc_edges = torch.stack([pc_to_gripper_dst, pc_to_gripper_src])
        data[self.EDGE_TYPE_GRIPPER_PC].edge_index = gripper_to_pc_edges
        gripper_pc_edge_features = compute_edge_features(
            gripper_positions, point_cloud, gripper_to_pc_edges,
            num_frequencies=self.num_edge_frequencies
        )
        data[self.EDGE_TYPE_GRIPPER_PC].edge_attr = self.edge_projection(gripper_pc_edge_features)
        
        # 4. Gripper internal edges (fully connected within the 6 nodes)
        gripper_src = []
        gripper_dst = []
        for i in range(self.num_gripper_nodes):
            for j in range(self.num_gripper_nodes):
                if i != j:
                    gripper_src.append(i)
                    gripper_dst.append(j)
        gripper_same_edges = torch.tensor([gripper_src, gripper_dst], device=device, dtype=torch.long)
        data[self.EDGE_TYPE_GRIPPER_GRIPPER_SAME].edge_index = gripper_same_edges
        gripper_same_edge_features = compute_edge_features(
            gripper_positions, gripper_positions, gripper_same_edges,
            num_frequencies=self.num_edge_frequencies
        )
        data[self.EDGE_TYPE_GRIPPER_GRIPPER_SAME].edge_attr = self.edge_projection(gripper_same_edge_features)
        
        return LocalGraph(
            data=data, 
            num_points=num_pc, 
            num_gripper_nodes=self.num_gripper_nodes
        )
    
    def _create_gripper_node_positions(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create 6 gripper node positions based on gripper pose.
        
        The 6 nodes represent the gripper structure:
        - 2 position nodes (fingertip positions)
        - 4 orientation nodes (gripper frame axes)
        
        Args:
            position: Gripper center position [3]
            rotation: Gripper rotation matrix [3, 3]
        
        Returns:
            Gripper node positions [6, 3]
        """
        device = position.device
        
        # Gripper geometry offsets (in gripper local frame)
        # These define the structure of the parallel-jaw gripper
        offsets = torch.tensor([
            [0.0, 0.04, 0.0],   # Left finger tip
            [0.0, -0.04, 0.0],  # Right finger tip
            [0.05, 0.0, 0.0],   # Forward (x-axis)
            [-0.03, 0.0, 0.0],  # Backward
            [0.0, 0.0, 0.03],   # Up (z-axis)
            [0.0, 0.0, -0.03],  # Down
        ], device=device, dtype=position.dtype)
        
        # Transform to world frame
        world_offsets = torch.matmul(offsets, rotation.T)  # [6, 3]
        positions = position.unsqueeze(0) + world_offsets  # [6, 3]
        
        return positions
    
    def build_context_graph(
        self,
        demo_graphs: List[List[LocalGraph]],
        live_graph: LocalGraph,
        noisy_ghost_positions: Optional[torch.Tensor] = None,
        diffusion_timestep: Optional[torch.Tensor] = None,
    ) -> ContextGraph:
        """
        Build a context graph aggregating demonstrations and current observation.
        
        Key design choices following the paper:
        1. TEMPORAL edges within each demo (t -> t+1), NOT all-to-all
        2. Demo gripper nodes connect to live gripper nodes
        3. Ghost gripper nodes (action hypotheses) connect to live gripper
        
        Args:
            demo_graphs: List of demonstrations, each containing list of LocalGraphs
                         Shape: [num_demos][num_waypoints]
            live_graph: Current observation LocalGraph
            noisy_ghost_positions: Noisy ghost gripper positions for diffusion [horizon, 6, 3]
            diffusion_timestep: Current diffusion timestep for embedding
        
        Returns:
            ContextGraph containing the aggregated hetero graph
        """
        data = HeteroData()
        device = live_graph.data[self.NODE_TYPE_POINT_CLOUD].x.device
        
        num_demos = len(demo_graphs)
        num_waypoints = len(demo_graphs[0]) if num_demos > 0 else 0
        
        # === Collect all nodes ===
        all_pc_features = []
        all_pc_pos = []
        all_gripper_features = []
        all_gripper_pos = []
        
        # Track node offsets
        pc_offset = 0
        gripper_offset = 0
        
        # Store gripper node indices for edge construction
        # demo_gripper_indices[demo_idx][waypoint_idx] = [node indices]
        demo_gripper_indices: List[List[List[int]]] = []
        
        # === Process demonstration graphs ===
        for demo_idx, demo in enumerate(demo_graphs):
            demo_gripper_idx_list = []
            
            for wp_idx, local_graph in enumerate(demo):
                g = local_graph.data
                
                # Accumulate point cloud nodes
                all_pc_features.append(g[self.NODE_TYPE_POINT_CLOUD].x)
                all_pc_pos.append(g[self.NODE_TYPE_POINT_CLOUD].pos)
                
                # Accumulate gripper nodes
                all_gripper_features.append(g[self.NODE_TYPE_GRIPPER].x)
                all_gripper_pos.append(g[self.NODE_TYPE_GRIPPER].pos)
                
                # Track gripper node indices
                gripper_indices = list(range(
                    gripper_offset, 
                    gripper_offset + self.num_gripper_nodes
                ))
                demo_gripper_idx_list.append(gripper_indices)
                
                # Update offsets
                pc_offset += g[self.NODE_TYPE_POINT_CLOUD].x.shape[0]
                gripper_offset += self.num_gripper_nodes
            
            demo_gripper_indices.append(demo_gripper_idx_list)
        
        # === Process live observation ===
        live_gripper_start = gripper_offset
        g = live_graph.data
        all_pc_features.append(g[self.NODE_TYPE_POINT_CLOUD].x)
        all_pc_pos.append(g[self.NODE_TYPE_POINT_CLOUD].pos)
        all_gripper_features.append(g[self.NODE_TYPE_GRIPPER].x)
        all_gripper_pos.append(g[self.NODE_TYPE_GRIPPER].pos)
        
        live_gripper_indices = list(range(
            gripper_offset, 
            gripper_offset + self.num_gripper_nodes
        ))
        gripper_offset += self.num_gripper_nodes
        
        # === Concatenate node features ===
        data[self.NODE_TYPE_POINT_CLOUD].x = torch.cat(all_pc_features, dim=0)
        data[self.NODE_TYPE_POINT_CLOUD].pos = torch.cat(all_pc_pos, dim=0)
        data[self.NODE_TYPE_GRIPPER].x = torch.cat(all_gripper_features, dim=0)
        data[self.NODE_TYPE_GRIPPER].pos = torch.cat(all_gripper_pos, dim=0)
        
        # === Build Temporal Edges (RED lines in paper Figure 2) ===
        # Connect t -> t+1 within each demonstration
        temporal_src = []
        temporal_dst = []
        
        for demo_idx, demo_gripper_idx_list in enumerate(demo_gripper_indices):
            for wp_idx in range(len(demo_gripper_idx_list) - 1):
                curr_indices = demo_gripper_idx_list[wp_idx]
                next_indices = demo_gripper_idx_list[wp_idx + 1]
                
                # Connect each gripper node at t to corresponding node at t+1
                for i in range(self.num_gripper_nodes):
                    temporal_src.append(curr_indices[i])
                    temporal_dst.append(next_indices[i])
        
        if temporal_src:
            temporal_edges = torch.tensor([temporal_src, temporal_dst], device=device, dtype=torch.long)
            data[self.EDGE_TYPE_TEMPORAL].edge_index = temporal_edges
            # Compute temporal edge features
            all_gripper_positions = data[self.NODE_TYPE_GRIPPER].pos
            temporal_edge_features = compute_edge_features(
                all_gripper_positions, all_gripper_positions, temporal_edges,
                num_frequencies=self.num_edge_frequencies
            )
            data[self.EDGE_TYPE_TEMPORAL].edge_attr = self.edge_projection(temporal_edge_features)
        
        # === Build Demo-to-Live Edges ===
        # Only connect the LAST waypoint of each demo to the live gripper
        demo_live_src = []
        demo_live_dst = []
        
        for demo_idx, demo_gripper_idx_list in enumerate(demo_gripper_indices):
            if len(demo_gripper_idx_list) > 0:
                # Last waypoint's gripper nodes
                last_wp_indices = demo_gripper_idx_list[-1]
                for src_idx in last_wp_indices:
                    for dst_idx in live_gripper_indices:
                        demo_live_src.append(src_idx)
                        demo_live_dst.append(dst_idx)
        
        if demo_live_src:
            demo_live_edges = torch.tensor([demo_live_src, demo_live_dst], device=device, dtype=torch.long)
            data[self.EDGE_TYPE_DEMO_LIVE].edge_index = demo_live_edges
            demo_live_edge_features = compute_edge_features(
                all_gripper_positions, all_gripper_positions, demo_live_edges,
                num_frequencies=self.num_edge_frequencies
            )
            data[self.EDGE_TYPE_DEMO_LIVE].edge_attr = self.edge_projection(demo_live_edge_features)
        
        # === Add Ghost Gripper Nodes (Action Hypotheses) ===
        num_ghost = 0
        if noisy_ghost_positions is not None:
            num_ghost = self._add_ghost_nodes(
                data, 
                noisy_ghost_positions,
                live_gripper_indices,
                diffusion_timestep,
            )
        
        return ContextGraph(
            data=data,
            num_demos=num_demos,
            num_waypoints_per_demo=num_waypoints,
            num_ghost_nodes=num_ghost,
        )
    
    def _add_ghost_nodes(
        self,
        data: HeteroData,
        noisy_positions: torch.Tensor,
        live_gripper_indices: List[int],
        diffusion_timestep: Optional[torch.Tensor],
    ) -> int:
        """
        Add ghost gripper nodes for action prediction.
        
        Ghost nodes represent future gripper keypoint hypotheses.
        They are connected to the live gripper nodes and to each other.
        
        Args:
            data: HeteroData to add nodes to
            noisy_positions: Noisy ghost positions [horizon, 6, 3] or [horizon, 3]
            live_gripper_indices: Indices of live gripper nodes
            diffusion_timestep: Current diffusion timestep
        
        Returns:
            Number of ghost nodes added
        """
        device = noisy_positions.device
        horizon = noisy_positions.shape[0]
        
        # Handle different input shapes
        if noisy_positions.dim() == 2:
            # [horizon, 3] - single position per timestep
            # Expand to 6 nodes using gripper structure
            ghost_positions = noisy_positions.unsqueeze(1).expand(-1, self.num_gripper_nodes, -1)
            ghost_positions = ghost_positions.reshape(-1, 3)  # [horizon * 6, 3]
            num_ghost_nodes = horizon * self.num_gripper_nodes
        else:
            # [horizon, 6, 3] - full gripper structure
            ghost_positions = noisy_positions.reshape(-1, 3)  # [horizon * 6, 3]
            num_ghost_nodes = horizon * self.num_gripper_nodes
        
        # Create ghost node features
        # 1. Position encoding
        pos_enc = sinusoidal_positional_encoding(
            ghost_positions,
            num_frequencies=self.num_pos_frequencies
        )
        pos_features = self.pos_projection(pos_enc)  # [num_ghost, feature_dim]
        
        # 2. Horizon/type embedding
        horizon_idx = torch.arange(horizon, device=device).repeat_interleave(self.num_gripper_nodes)
        horizon_emb = self.ghost_type_embedding(horizon_idx % self.prediction_horizon)
        
        # 3. Diffusion timestep embedding (if provided)
        if diffusion_timestep is not None:
            t_emb = self._get_sinusoidal_timestep_embedding(diffusion_timestep)
            t_emb = t_emb.expand(num_ghost_nodes, -1)
            ghost_features = pos_features + horizon_emb + t_emb
        else:
            ghost_features = pos_features + horizon_emb
        
        data[self.NODE_TYPE_GHOST].x = ghost_features
        data[self.NODE_TYPE_GHOST].pos = ghost_positions
        
        # === Build Live-to-Ghost Edges ===
        live_ghost_src = []
        live_ghost_dst = []
        for live_idx in live_gripper_indices:
            for ghost_idx in range(num_ghost_nodes):
                live_ghost_src.append(live_idx)
                live_ghost_dst.append(ghost_idx)
        
        live_ghost_edges = torch.tensor([live_ghost_src, live_ghost_dst], device=device, dtype=torch.long)
        data[self.EDGE_TYPE_LIVE_GHOST].edge_index = live_ghost_edges
        
        all_gripper_pos = data[self.NODE_TYPE_GRIPPER].pos
        live_ghost_edge_features = compute_edge_features(
            all_gripper_pos, ghost_positions, live_ghost_edges,
            num_frequencies=self.num_edge_frequencies
        )
        data[self.EDGE_TYPE_LIVE_GHOST].edge_attr = self.edge_projection(live_ghost_edge_features)
        
        # === Build Ghost-to-Live Edges (reverse) ===
        ghost_live_edges = torch.tensor([live_ghost_dst, live_ghost_src], device=device, dtype=torch.long)
        data[self.EDGE_TYPE_GHOST_LIVE].edge_index = ghost_live_edges
        ghost_live_edge_features = compute_edge_features(
            ghost_positions, all_gripper_pos, ghost_live_edges,
            num_frequencies=self.num_edge_frequencies
        )
        data[self.EDGE_TYPE_GHOST_LIVE].edge_attr = self.edge_projection(ghost_live_edge_features)
        
        # === Build Ghost-to-Ghost Edges (sequential) ===
        # Connect ghost nodes from t to t+1
        ghost_ghost_src = []
        ghost_ghost_dst = []
        
        for t in range(horizon - 1):
            for i in range(self.num_gripper_nodes):
                curr_idx = t * self.num_gripper_nodes + i
                next_idx = (t + 1) * self.num_gripper_nodes + i
                ghost_ghost_src.append(curr_idx)
                ghost_ghost_dst.append(next_idx)
                # Also add reverse
                ghost_ghost_src.append(next_idx)
                ghost_ghost_dst.append(curr_idx)
        
        if ghost_ghost_src:
            ghost_ghost_edges = torch.tensor([ghost_ghost_src, ghost_ghost_dst], device=device, dtype=torch.long)
            data[self.EDGE_TYPE_GHOST_GHOST].edge_index = ghost_ghost_edges
            ghost_ghost_edge_features = compute_edge_features(
                ghost_positions, ghost_positions, ghost_ghost_edges,
                num_frequencies=self.num_edge_frequencies
            )
            data[self.EDGE_TYPE_GHOST_GHOST].edge_attr = self.edge_projection(ghost_ghost_edge_features)
        
        return num_ghost_nodes
    
    def create_initial_ghost_positions(
        self,
        live_gripper_pos: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """
        Create initial (noisy) ghost positions for diffusion.
        
        Starts from random noise around the live gripper position.
        
        Args:
            live_gripper_pos: Live gripper positions [6, 3]
            horizon: Prediction horizon
        
        Returns:
            Initial ghost positions [horizon, 6, 3]
        """
        device = live_gripper_pos.device
        
        # Start from live position with added noise
        # The noise represents the initial random state for diffusion
        ghost_positions = live_gripper_pos.unsqueeze(0).expand(horizon, -1, -1).clone()
        noise = torch.randn_like(ghost_positions) * 0.1  # Small initial noise
        ghost_positions = ghost_positions + noise
        
        return ghost_positions
