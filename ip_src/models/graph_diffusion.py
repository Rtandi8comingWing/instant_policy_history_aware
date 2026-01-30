"""
GraphDiffusion - Main model for Instant Policy.

This module implements the complete Instant Policy model following the paper:
- Graph-based representation with edge features
- Three-stage network (σ, φ, ψ) with edge-aware attention
- Flow matching for geometric action generation in SE(3) space
- Ghost gripper nodes as action hypotheses

Key implementation details (following Appendix B):
1. Uses GRAPH structure throughout (not pooled features)
2. Edge features participate in attention (Equation 3)
3. Noise is added in se(3) (tangent space), NOT Euclidean coordinates
4. Logmap/Expmap for SE(3) <-> se(3) conversion
5. Normalization to [-1, 1] with 1cm translation and 3° rotation scales
6. SVD (Arun's Method) for SE(3) recovery from point displacements

Compatible with the original instant_policy.pyi interface.

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from torch_geometric.data import Batch, HeteroData

from ip_src.models.encoders import PointNetPlusPlusEncoder
from ip_src.models.graph_transformer import (
    LocalGraphEncoder,
    ContextAggregator,
    ActionDecoder,
)
from ip_src.models.diffusion import (
    FlowMatchingScheduler,
    FlowMatchingConfig,
)
from ip_src.data.graph_builder import GraphBuilder, LocalGraph, ContextGraph
from ip_src.utils.se3_utils import (
    se3_log_map,
    se3_exp_map,
    SE3ActionNormalizer,
    compute_relative_transform,
    apply_transform_to_points,
    poses_to_gripper_positions,
    DEFAULT_GRIPPER_OFFSETS,
)


def svd_se3_recovery(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> torch.Tensor:
    """
    Recover SE(3) transformation from point correspondences using SVD (Arun's Method).
    
    Given source points P and target points Q, finds the optimal rotation R and
    translation t such that Q ≈ R @ P + t
    
    Reference: Arun, K. S., Huang, T. S., & Blostein, S. D. (1987).
    "Least-squares fitting of two 3-D point sets."
    
    Args:
        source_points: Source point positions [N, 3]
        target_points: Target point positions [N, 3]
    
    Returns:
        SE(3) transformation matrix [4, 4]
    """
    assert source_points.shape == target_points.shape
    assert source_points.shape[-1] == 3
    
    # Compute centroids
    centroid_src = source_points.mean(dim=0)  # [3]
    centroid_tgt = target_points.mean(dim=0)  # [3]
    
    # Center the points
    src_centered = source_points - centroid_src  # [N, 3]
    tgt_centered = target_points - centroid_tgt  # [N, 3]
    
    # Compute covariance matrix H = Σ (src_i) (tgt_i)^T
    H = src_centered.T @ tgt_centered  # [3, 3]
    
    # SVD: H = U @ S @ V^T
    U, S, Vt = torch.linalg.svd(H)
    
    # Rotation: R = V @ U^T
    R = Vt.T @ U.T
    
    # Handle reflection case (det(R) = -1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Translation: t = centroid_tgt - R @ centroid_src
    t = centroid_tgt - R @ centroid_src
    
    # Build SE(3) matrix
    T = torch.eye(4, device=source_points.device, dtype=source_points.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


class GraphDiffusion(pl.LightningModule):
    """
    GraphDiffusion: In-Context Imitation Learning via Graph Diffusion.
    
    This implements the paper's approach:
    1. Build graph representations with edge features
    2. Encode local graphs with σ(·)
    3. Aggregate context with φ(·) using temporal edges
    4. Predict geometric flow with ψ(·) on ghost gripper nodes
    5. Iteratively refine ghost positions via flow matching
    6. Recover SE(3) via SVD from predicted point displacements
    
    Interface compatible with instant_policy.pyi.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        hidden_dim: int = 256,
        edge_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_points: int = 16,
        k_neighbors: int = 8,
        num_gripper_nodes: int = 6,
        prediction_horizon: int = 8,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 4,
        num_pos_frequencies: int = 10,
        num_edge_frequencies: int = 6,
        freeze_geometry_encoder: bool = True,
        encoder_checkpoint: Optional[str] = None,
        encoder_source_prefix: str = "scene_encoder.",
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
    ):
        """
        Initialize GraphDiffusion model.
        
        Args:
            device: Device to run on
            hidden_dim: Hidden dimension for transformers (and geometry encoder output)
            edge_dim: Edge feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            num_points: Number of keypoint nodes M from geometry encoder (default: 16)
            k_neighbors: k for k-NN in graph (should be < num_points)
            num_gripper_nodes: Gripper nodes per timestep (6)
            prediction_horizon: Future keypoints to predict
            num_train_timesteps: Flow matching training steps
            num_inference_timesteps: Flow matching inference steps
            num_pos_frequencies: Position encoding frequencies
            num_edge_frequencies: Edge encoding frequencies
            freeze_geometry_encoder: Whether to freeze PointNet++ (recommended: True)
            encoder_checkpoint: Path to pretrained model.pt for loading encoder weights
            encoder_source_prefix: Key prefix in checkpoint for encoder (default: "scene_encoder.")
            learning_rate: Learning rate (paper uses 1e-5)
            weight_decay: Weight decay for AdamW
        """
        super().__init__()
        self.save_hyperparameters()
        
        self._device = device
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_points = num_points
        self.num_gripper_nodes = num_gripper_nodes
        self.prediction_horizon = prediction_horizon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Runtime parameters
        self._num_demos = 2
        self._num_diffusion_steps = num_inference_timesteps
        
        # === Model Components ===
        
        # 1. Geometry Encoder (frozen PointNet++)
        # Paper Appendix A: encoder outputs 512-dim features (fixed to match pretrained weights)
        self.geometry_encoder = PointNetPlusPlusEncoder(
            freeze=freeze_geometry_encoder,
        )
        
        # Load pretrained encoder weights if checkpoint provided
        if encoder_checkpoint is not None:
            print(f"Loading pretrained encoder weights from: {encoder_checkpoint}")
            self.geometry_encoder.load_pretrained_weights(
                encoder_checkpoint,
                source_prefix=encoder_source_prefix,
                strict=False,
                verbose=True,
            )
        
        # Projection layer: 512 (encoder output) -> hidden_dim (graph features)
        # This allows using pretrained encoder with any hidden_dim
        self.encoder_projection = nn.Linear(512, hidden_dim)
        
        # 2. Graph Builder with edge features
        self.graph_builder = GraphBuilder(
            num_points=num_points,
            k_neighbors=k_neighbors,
            num_gripper_nodes=num_gripper_nodes,
            feature_dim=hidden_dim,
            num_pos_frequencies=num_pos_frequencies,
            num_edge_frequencies=num_edge_frequencies,
            prediction_horizon=prediction_horizon,
            device=device,
        )
        
        # 3. Local Graph Encoder σ(·)
        self.local_encoder = LocalGraphEncoder(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # 4. Context Aggregator φ(·)
        self.context_aggregator = ContextAggregator(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # 5. Action Decoder ψ(·) - Graph Transformer
        self.action_decoder = ActionDecoder(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            flow_dim=6,  # 3D translation + 3D rotation flow
        )
        
        # 6. Flow Matching Scheduler
        flow_config = FlowMatchingConfig(
            num_train_timesteps=num_train_timesteps,
            num_inference_timesteps=num_inference_timesteps,
        )
        self.flow_scheduler = FlowMatchingScheduler(flow_config)
        
        # 7. SE(3) Action Normalizer (Appendix B & E)
        # Translation scale: 1cm, Rotation scale: 3 degrees ≈ 0.0524 radians
        self.action_normalizer = SE3ActionNormalizer(
            trans_scale=0.01,  # 1cm
            rot_scale=0.0524,  # ~3 degrees
        )
        
        # Register gripper offsets as buffer
        self.register_buffer("gripper_offsets", DEFAULT_GRIPPER_OFFSETS)
    
    def set_num_demos(self, num_demos: int) -> None:
        """Set number of demonstrations to use."""
        self._num_demos = num_demos
    
    def set_num_diffusion_steps(self, num_diffusion_steps: int) -> None:
        """Set number of diffusion/flow matching steps."""
        self._num_diffusion_steps = num_diffusion_steps
        self.flow_scheduler.set_num_inference_steps(num_diffusion_steps)
    
    def _extract_graph_data(self, data: HeteroData) -> Tuple[Dict, Dict, Dict]:
        """Extract node features, edge indices, and edge attributes from HeteroData."""
        x_dict = {}
        edge_index_dict = {}
        edge_attr_dict = {}
        
        for node_type in data.node_types:
            x_dict[node_type] = data[node_type].x
        
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                edge_attr_dict[edge_type] = data[edge_type].edge_attr
        
        return x_dict, edge_index_dict, edge_attr_dict
    
    def _encode_local_graph(self, local_graph: LocalGraph) -> Dict[str, torch.Tensor]:
        """Encode a local graph with σ(·)."""
        x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(local_graph.data)
        return self.local_encoder(x_dict, edge_index_dict, edge_attr_dict)
    
    def _build_single_sample_graphs(
        self,
        demo_pcds: torch.Tensor,  # [num_demos, num_wp, num_points, 3]
        demo_poses: torch.Tensor,  # [num_demos, num_wp, 4, 4]
        demo_grips: torch.Tensor,  # [num_demos, num_wp]
        live_pcd: torch.Tensor,    # [num_points, 3]
        live_pose: torch.Tensor,   # [4, 4]
        live_grip: torch.Tensor,   # scalar
    ) -> Tuple[List[List[LocalGraph]], LocalGraph]:
        """
        Build demo and live graphs for a single sample.
        
        This properly uses graph_builder, local_encoder for training.
        """
        device = live_pcd.device
        num_demos = demo_pcds.shape[0]
        num_wp = demo_pcds.shape[1]
        
        # === Build Demo Graphs ===
        demo_graphs = []
        for d in range(num_demos):
            waypoint_graphs = []
            for w in range(num_wp):
                pcd = demo_pcds[d, w]
                pose = demo_poses[d, w]
                grip = demo_grips[d, w]
                
                # Encode point cloud with PointNet++
                # Returns M=16 keypoint features (512-dim) and positions
                point_features, point_positions = self.geometry_encoder(pcd)
                point_features = self.encoder_projection(point_features)  # 512 -> hidden_dim
                
                # Build local graph using keypoint positions (not raw point cloud)
                local_graph = self.graph_builder.build_local_graph(
                    point_cloud=point_positions,
                    point_features=point_features,
                    gripper_pose=pose,
                    gripper_state=grip.unsqueeze(0) if grip.dim() == 0 else grip,
                )
                
                # Encode local graph with σ(·)
                encoded = self._encode_local_graph(local_graph)
                local_graph.data['point_cloud'].x = encoded['point_cloud']
                local_graph.data['gripper'].x = encoded['gripper']
                
                waypoint_graphs.append(local_graph)
            demo_graphs.append(waypoint_graphs)
        
        # === Build Live Graph ===
        # Encode point cloud - returns M=16 keypoint features (512-dim) and positions
        point_features, point_positions = self.geometry_encoder(live_pcd)
        point_features = self.encoder_projection(point_features)  # 512 -> hidden_dim
        live_graph = self.graph_builder.build_local_graph(
            point_cloud=point_positions,
            point_features=point_features,
            gripper_pose=live_pose,
            gripper_state=live_grip.unsqueeze(0) if live_grip.dim() == 0 else live_grip,
        )
        encoded = self._encode_local_graph(live_graph)
        live_graph.data['point_cloud'].x = encoded['point_cloud']
        live_graph.data['gripper'].x = encoded['gripper']
        
        return demo_graphs, live_graph
    
    def forward(
        self,
        demo_pcds: torch.Tensor,
        demo_poses: torch.Tensor,
        demo_grips: torch.Tensor,
        live_pcd: torch.Tensor,
        live_pose: torch.Tensor,
        live_grip: torch.Tensor,
        target_poses: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
        target_grips: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with SE(3) flow matching (Appendix B).
        
        Following the paper's noise addition procedure:
        1. Compute ground truth action T_EA = T_WE^{-1} @ T_WA (relative SE(3))
        2. Project to se(3) via Logmap
        3. Normalize to [-1, 1] using action_scale (1cm trans, 3° rot)
        4. Add flow matching noise in normalized se(3) space
        5. Unnormalize and project back to SE(3) via Expmap
        6. Apply noisy action to live gripper to get noisy ghost positions
        7. Predict flow in normalized se(3) space
        
        Args:
            demo_pcds: Demo point clouds [B, num_demos, num_wp, num_points, 3]
            demo_poses: Demo poses [B, num_demos, num_wp, 4, 4]
            demo_grips: Demo gripper states [B, num_demos, num_wp]
            live_pcd: Live point cloud [B, num_points, 3]
            live_pose: Live pose [B, 4, 4] (T_WE - current end-effector pose)
            live_grip: Live gripper state [B]
            target_poses: Target poses [B, horizon, 4, 4] (T_WA - target poses)
            target_positions: Target ghost positions [B, horizon, 6, 3] (optional, for logging)
            target_grips: Target gripper states [B, horizon] (optional)
        
        Returns:
            Dictionary with loss components
        """
        B = live_pcd.shape[0]
        H = target_poses.shape[1]  # horizon
        device = live_pcd.device
        
        # Process each sample and collect context graphs for batching
        all_context_graphs = []
        all_target_flows = []  # Target flow in normalized se(3) space
        all_timesteps = []
        
        for b in range(B):
            # Build demo and live graphs using proper graph pipeline
            demo_graphs, live_graph = self._build_single_sample_graphs(
                demo_pcds=demo_pcds[b],
                demo_poses=demo_poses[b],
                demo_grips=demo_grips[b],
                live_pcd=live_pcd[b],
                live_pose=live_pose[b],
                live_grip=live_grip[b],
            )
            
            # Sample timestep for flow matching (uniform in [0, 1])
            t = torch.rand(1, device=device)
            all_timesteps.append(t)
            
            # === Step A: Compute Ground Truth SE(3) Actions ===
            # T_EA = T_WE^{-1} @ T_WA for each horizon step
            live_T_WE = live_pose[b]  # [4, 4]
            target_T_WAs = target_poses[b]  # [horizon, 4, 4]
            
            # For each horizon step, compute relative action from current pose
            # Note: We compute action relative to LIVE pose, not previous step
            gt_actions_se3 = []  # [horizon, 4, 4]
            for h in range(H):
                T_WA = target_T_WAs[h]
                T_EA = compute_relative_transform(T_WA.unsqueeze(0), live_T_WE.unsqueeze(0)).squeeze(0)
                gt_actions_se3.append(T_EA)
            gt_actions_se3 = torch.stack(gt_actions_se3)  # [horizon, 4, 4]
            
            # === Step B: Project to se(3) via Logmap ===
            gt_xi = se3_log_map(gt_actions_se3)  # [horizon, 6]
            
            # === Step C: Normalize to [-1, 1] ===
            gt_xi_norm = self.action_normalizer.normalize(gt_xi)  # [horizon, 6]
            
            # === Step D: Flow Matching Noise in Normalized se(3) Space ===
            # Sample noise in normalized space (standard Gaussian)
            noise_xi_norm = torch.randn_like(gt_xi_norm)  # [horizon, 6]
            
            # Interpolate: xi_t = (1-t)*gt_xi_norm + t*noise_xi_norm
            noisy_xi_norm = self.flow_scheduler.interpolate(gt_xi_norm, noise_xi_norm, t)
            
            # Target velocity (flow): v = noise - target
            target_flow = self.flow_scheduler.get_velocity(gt_xi_norm, noise_xi_norm, t)
            all_target_flows.append(target_flow)  # [horizon, 6]
            
            # === Step E: Unnormalize and Expmap to SE(3) ===
            noisy_xi = self.action_normalizer.unnormalize(noisy_xi_norm)  # [horizon, 6]
            noisy_actions_se3 = se3_exp_map(noisy_xi)  # [horizon, 4, 4]
            
            # === Step F: Apply Noisy Actions to Live Gripper ===
            # Get live gripper positions
            live_gripper_pos = live_graph.data['gripper'].pos  # [6, 3]
            
            # Apply each noisy action to live gripper
            noisy_ghost_positions = []
            for h in range(H):
                # Transform live gripper points by noisy action
                transformed = apply_transform_to_points(
                    live_gripper_pos.unsqueeze(0),  # [1, 6, 3]
                    noisy_actions_se3[h].unsqueeze(0),  # [1, 4, 4]
                ).squeeze(0)  # [6, 3]
                noisy_ghost_positions.append(transformed)
            noisy_ghost_positions = torch.stack(noisy_ghost_positions)  # [horizon, 6, 3]
            
            # === Step G: Build Context Graph with Noisy Ghost Positions ===
            context_graph = self.graph_builder.build_context_graph(
                demo_graphs=demo_graphs,
                live_graph=live_graph,
                noisy_ghost_positions=noisy_ghost_positions,
                diffusion_timestep=t,
            )
            
            all_context_graphs.append(context_graph.data)
        
        # === Batch the context graphs using PyG Batch ===
        batched_graph = Batch.from_data_list(all_context_graphs)
        
        # === Aggregate context with φ(·) ===
        x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(batched_graph)
        context_out = self.context_aggregator(x_dict, edge_index_dict, edge_attr_dict)
        
        # Update batched graph with aggregated features
        for node_type, features in context_out.items():
            batched_graph[node_type].x = features
        
        # === Predict flow with ψ(·) ===
        # Output: 6D flow per ghost node (3 trans + 3 rot in normalized se(3))
        x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(batched_graph)
        pred_flow, grip_pred = self.action_decoder(x_dict, edge_index_dict, edge_attr_dict)
        
        # === Unbatch predictions ===
        num_ghost_per_sample = self.prediction_horizon * self.num_gripper_nodes
        
        pred_flows = []
        grip_preds = []
        
        if 'ghost' in batched_graph.node_types:
            ghost_batch = batched_graph['ghost'].batch
            
            for b in range(B):
                mask = ghost_batch == b
                sample_flow = pred_flow[mask]  # [num_ghost_per_sample, 6]
                sample_grip = grip_pred[mask]  # [num_ghost_per_sample, 1]
                
                # Reshape to [horizon, 6, 6] and [horizon, 6, 1]
                sample_flow = sample_flow.view(
                    self.prediction_horizon, self.num_gripper_nodes, -1
                )
                sample_grip = sample_grip.view(
                    self.prediction_horizon, self.num_gripper_nodes, -1
                )
                
                pred_flows.append(sample_flow)
                grip_preds.append(sample_grip)
        else:
            for b in range(B):
                start_idx = b * num_ghost_per_sample
                end_idx = start_idx + num_ghost_per_sample
                
                sample_flow = pred_flow[start_idx:end_idx]
                sample_grip = grip_pred[start_idx:end_idx]
                
                sample_flow = sample_flow.view(
                    self.prediction_horizon, self.num_gripper_nodes, -1
                )
                sample_grip = sample_grip.view(
                    self.prediction_horizon, self.num_gripper_nodes, -1
                )
                
                pred_flows.append(sample_flow)
                grip_preds.append(sample_grip)
        
        # Stack batch
        pred_flows = torch.stack(pred_flows)  # [B, horizon, 6, 6]
        grip_preds = torch.stack(grip_preds)  # [B, horizon, 6, 1]
        target_flows = torch.stack(all_target_flows)  # [B, horizon, 6]
        
        # Average flow prediction across 6 gripper nodes
        # Each node predicts the same SE(3) flow, so we average
        pred_flow_avg = pred_flows.mean(dim=2)  # [B, horizon, 6]
        
        return {
            'pred_flow': pred_flow_avg,  # [B, horizon, 6] - normalized se(3) flow
            'target_flow': target_flows,  # [B, horizon, 6] - normalized se(3) flow
            'pred_grip': grip_preds,
            'target_grip': target_grips,
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        PyTorch Lightning training step.
        
        Loss is computed in normalized se(3) space (6D flow vectors).
        """
        outputs = self.forward(
            demo_pcds=batch['demo_pcds'],
            demo_poses=batch['demo_poses'],
            demo_grips=batch['demo_grips'],
            live_pcd=batch['live_pcd'],
            live_pose=batch['live_pose'],
            live_grip=batch['live_grip'],
            target_poses=batch['target_poses'],
            target_positions=batch.get('target_positions'),
            target_grips=batch.get('target_grips'),
        )
        
        # Flow matching loss in normalized se(3) space
        # Both pred_flow and target_flow are [B, horizon, 6]
        flow_loss = F.mse_loss(outputs['pred_flow'], outputs['target_flow'])
        
        # Split into translation and rotation components for logging
        trans_loss = F.mse_loss(outputs['pred_flow'][..., :3], outputs['target_flow'][..., :3])
        rot_loss = F.mse_loss(outputs['pred_flow'][..., 3:], outputs['target_flow'][..., 3:])
        
        # Gripper prediction loss (BCE)
        if outputs['target_grip'] is not None:
            # Average gripper prediction across gripper nodes
            pred_grip_avg = outputs['pred_grip'].mean(dim=2)  # [B, horizon, 1]
            target_grip = outputs['target_grip'].unsqueeze(-1).float()  # [B, horizon, 1]
            grip_loss = F.binary_cross_entropy_with_logits(pred_grip_avg, target_grip)
        else:
            grip_loss = torch.tensor(0.0, device=self.device)
        
        # Total loss
        loss = flow_loss + 0.1 * grip_loss
        
        # Logging
        self.log('train/flow_loss', flow_loss, prog_bar=True)
        self.log('train/trans_loss', trans_loss)
        self.log('train/rot_loss', rot_loss)
        self.log('train/grip_loss', grip_loss)
        self.log('train/total_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """PyTorch Lightning validation step."""
        outputs = self.forward(
            demo_pcds=batch['demo_pcds'],
            demo_poses=batch['demo_poses'],
            demo_grips=batch['demo_grips'],
            live_pcd=batch['live_pcd'],
            live_pose=batch['live_pose'],
            live_grip=batch['live_grip'],
            target_poses=batch['target_poses'],
            target_positions=batch.get('target_positions'),
            target_grips=batch.get('target_grips'),
        )
        
        flow_loss = F.mse_loss(outputs['pred_flow'], outputs['target_flow'])
        trans_loss = F.mse_loss(outputs['pred_flow'][..., :3], outputs['target_flow'][..., :3])
        rot_loss = F.mse_loss(outputs['pred_flow'][..., 3:], outputs['target_flow'][..., 3:])
        
        self.log('val/flow_loss', flow_loss, prog_bar=True)
        self.log('val/trans_loss', trans_loss)
        self.log('val/rot_loss', rot_loss)
        
        return flow_loss
    
    def configure_optimizers(self):
        """Configure optimizer (AdamW as per paper Appendix H)."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100000,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    @torch.no_grad()
    def predict_actions(self, full_sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict actions given demonstrations and current observation.
        
        Uses iterative flow matching in SE(3) tangent space (se(3)):
        1. Initialize with noise in normalized se(3) space
        2. Iteratively denoise using predicted flow
        3. Convert final se(3) actions to SE(3) matrices via Expmap
        
        Compatible with original instant_policy.pyi interface.
        
        Args:
            full_sample: Dictionary containing:
                - 'demos': List of N demo dictionaries
                - 'live': Current observation dictionary
        
        Returns:
            actions: Predicted transforms [horizon, 4, 4] (T_EA - action in EE frame)
            grips: Predicted gripper states [horizon, 1]
        """
        self.eval()
        device = next(self.parameters()).device
        
        # === Process Demonstrations ===
        demo_graphs = []
        for demo in full_sample['demos'][:self._num_demos]:
            waypoint_graphs = []
            for t in range(len(demo['obs'])):
                # Get observation data
                pcd = torch.tensor(demo['obs'][t], dtype=torch.float32, device=device)
                T_w_e = torch.tensor(demo['T_w_es'][t], dtype=torch.float32, device=device)
                grip = torch.tensor([demo['grips'][t]], dtype=torch.float32, device=device)
                
                # Encode point cloud - returns M=16 keypoint features (512-dim) and positions
                point_features, point_positions = self.geometry_encoder(pcd)
                point_features = self.encoder_projection(point_features)  # 512 -> hidden_dim
                
                # Build local graph using keypoint positions
                local_graph = self.graph_builder.build_local_graph(
                    point_cloud=point_positions,
                    point_features=point_features,
                    gripper_pose=T_w_e,
                    gripper_state=grip,
                )
                
                # Encode local graph
                encoded = self._encode_local_graph(local_graph)
                
                # Update graph with encoded features
                local_graph.data['point_cloud'].x = encoded['point_cloud']
                local_graph.data['gripper'].x = encoded['gripper']
                
                waypoint_graphs.append(local_graph)
            
            demo_graphs.append(waypoint_graphs)
        
        # === Process Live Observation ===
        live_obs = full_sample['live']
        pcd = torch.tensor(live_obs['obs'][0], dtype=torch.float32, device=device)
        T_w_e = torch.tensor(live_obs['T_w_es'][0], dtype=torch.float32, device=device)
        grip = torch.tensor([live_obs['grips'][0]], dtype=torch.float32, device=device)
        
        # Store live pose for reference
        live_T_WE = T_w_e.clone()  # [4, 4]
        
        # Encode live point cloud - returns M=16 keypoint features (512-dim) and positions
        point_features, point_positions = self.geometry_encoder(pcd)
        point_features = self.encoder_projection(point_features)  # 512 -> hidden_dim
        
        # Build live local graph using keypoint positions
        live_graph = self.graph_builder.build_local_graph(
            point_cloud=point_positions,
            point_features=point_features,
            gripper_pose=T_w_e,
            gripper_state=grip,
        )
        
        # Encode live local graph
        encoded = self._encode_local_graph(live_graph)
        live_graph.data['point_cloud'].x = encoded['point_cloud']
        live_graph.data['gripper'].x = encoded['gripper']
        
        # Get live gripper positions for reference
        live_gripper_pos = live_graph.data['gripper'].pos.clone()  # [6, 3]
        
        # === Initialize SE(3) Actions from Noise ===
        # Start with random normalized se(3) vectors (like Gaussian noise)
        current_xi_norm = torch.randn(
            self.prediction_horizon, 6, device=device
        )  # [horizon, 6]
        
        # === Iterative Flow Matching Denoising ===
        timesteps = self.flow_scheduler.inference_timesteps
        step_sizes = self.flow_scheduler.step_sizes
        
        gripper_preds = []
        
        for i, (t, dt) in enumerate(zip(timesteps, step_sizes)):
            # Convert current normalized se(3) to SE(3) actions
            current_xi = self.action_normalizer.unnormalize(current_xi_norm)  # [horizon, 6]
            current_actions_se3 = se3_exp_map(current_xi)  # [horizon, 4, 4]
            
            # Apply current noisy actions to live gripper to get ghost positions
            ghost_positions = []
            for h in range(self.prediction_horizon):
                transformed = apply_transform_to_points(
                    live_gripper_pos.unsqueeze(0),  # [1, 6, 3]
                    current_actions_se3[h].unsqueeze(0),  # [1, 4, 4]
                ).squeeze(0)  # [6, 3]
                ghost_positions.append(transformed)
            ghost_positions = torch.stack(ghost_positions)  # [horizon, 6, 3]
            
            # Build context graph with current ghost positions
            context_graph = self.graph_builder.build_context_graph(
                demo_graphs=demo_graphs,
                live_graph=live_graph,
                noisy_ghost_positions=ghost_positions,
                diffusion_timestep=torch.tensor([t], device=device),
            )
            
            # Aggregate context with φ(·)
            x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(context_graph.data)
            context_out = self.context_aggregator(x_dict, edge_index_dict, edge_attr_dict)
            
            # Update graph with aggregated features
            for node_type, features in context_out.items():
                context_graph.data[node_type].x = features
            
            # Predict flow with ψ(·)
            # Output: [num_ghost, 6] - 6D flow in normalized se(3) space
            x_dict, edge_index_dict, edge_attr_dict = self._extract_graph_data(context_graph.data)
            flow, grip_pred = self.action_decoder(x_dict, edge_index_dict, edge_attr_dict)
            
            gripper_preds.append(grip_pred)
            
            # Reshape flow to [horizon, 6, 6] (6 gripper nodes, 6D flow each)
            flow = flow.view(self.prediction_horizon, self.num_gripper_nodes, 6)
            
            # Average flow across gripper nodes
            flow_avg = flow.mean(dim=1)  # [horizon, 6]
            
            # Integration step in normalized se(3) space: xi_{t-dt} = xi_t - dt * v
            current_xi_norm = current_xi_norm - dt.item() * flow_avg
        
        # === Convert Final se(3) to SE(3) Actions ===
        final_xi = self.action_normalizer.unnormalize(current_xi_norm)  # [horizon, 6]
        final_actions_se3 = se3_exp_map(final_xi)  # [horizon, 4, 4]
        
        # These are T_EA (action in EE frame) - transforms relative to live pose
        # To get absolute world poses: T_WA = T_WE @ T_EA
        actions = final_actions_se3  # [horizon, 4, 4]
        
        # === Get Gripper Predictions ===
        final_grip = gripper_preds[-1].view(
            self.prediction_horizon, self.num_gripper_nodes, 1
        )
        # Average across gripper nodes and threshold
        grips = torch.sigmoid(final_grip.mean(dim=1))  # [horizon, 1]
        grips = (grips > 0.5).float()
        
        return actions.cpu().numpy(), grips.cpu().numpy()
    
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        strict: bool = True,
        map_location: Optional[str] = None,
        **kwargs,
    ) -> "GraphDiffusion":
        """
        Load model from checkpoint.
        
        Compatible with original interface.
        """
        if map_location is None:
            map_location = device
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Extract hyperparameters
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
            model = cls(**hparams)
        else:
            model = cls(device=device, **kwargs)
        
        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)
        
        model = model.to(device)
        model.eval()
        return model
    
    def load_encoder_weights(
        self,
        checkpoint_path: str,
        source_prefix: str = "scene_encoder.",
        strict: bool = False,
    ):
        """
        Load pretrained encoder weights after model initialization.
        
        Args:
            checkpoint_path: Path to model.pt
            source_prefix: Key prefix in checkpoint
            strict: Whether to require exact key match
        
        Example:
            model = GraphDiffusion(device="cuda")
            model.load_encoder_weights("./model.pt")
        """
        self.geometry_encoder.load_pretrained_weights(
            checkpoint_path,
            source_prefix=source_prefix,
            strict=strict,
            verbose=True,
        )
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'state_dict': self.state_dict(),
            'hyper_parameters': dict(self.hparams),
        }
        torch.save(checkpoint, path)
