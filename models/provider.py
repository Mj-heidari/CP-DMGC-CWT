from models.EEGNet import EEGNet
from models.CE_stSENet.CE_stSENet import CE_stSENet
from models.cspnet import CSPNet
from models.stnet import STNet
from models.simplevit import SimpleViT
from models.TSception import TSception
from models.FBMSNet import FBMSNet
from models.labram import LaBraM
from models.rgnn import RGNN_Model
from models.dgcnn2 import DGCNN_Model
from models.dgcnn import DGCNN
from models.conformer import Conformer
from models.TSLANet import TSLANet
from models.LMDA import LMDA
from models.MB_dMGC_CWTFFNet import MB_dMGC_CWTFFNet
from models.EEGBandClassifier import EEGBandClassifier
from models.EEG_GNN_SSL import DCRNNModel_classification, get_adjacency_matrix
import torch.nn as nn
import torch
from functools import partial


channels = [
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
]


def initialize_edge_weights(num_nodes: int, seed: int = 42, diag_value: float = 1.0):
    """
    Initializes edge_index and edge_weight matrices for a fully connected graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes (EEG channels, etc.)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    diag_value : float, optional
        Value to fill on the diagonal (e.g., 1.0 for self-loops)

    Returns
    -------
    edge_index : torch.LongTensor
        Edge indices of shape [2, num_nodes * num_nodes]
    edge_weight : torch.FloatTensor
        Flattened edge weights of shape [num_nodes * num_nodes]
    """

    # Make initialization reproducible
    torch.manual_seed(seed)

    # Build fully connected edge index (including self-loops)
    row, col = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
    )
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)

    # Initialize weights (num_nodes x num_nodes)
    edge_weight = torch.empty(num_nodes, num_nodes)
    torch.nn.init.xavier_normal_(edge_weight)  # Xavier initialization

    # Set diagonal (self-loop weights)
    edge_weight.fill_diagonal_(diag_value)

    # Flatten to match model input
    edge_weight = edge_weight.reshape(-1)

    return edge_index, edge_weight


def model_builder(model_class, **kwargs):
    """
    Returns a function that builds a fresh model instance.
    This avoids weight leakage across folds.

    Args:
        model_class: torch.nn.Module class (e.g., EEGNet)
        kwargs: parameters to initialize the model

    Returns:
        A callable that builds a new model each time it's called
    """

    def build():
        return model_class(**kwargs)

    return build


def get_builder(model: str = "CE-stSENet"):
    match model:
        case "EEGNET":
            builder = model_builder(
                EEGNet,
                chunk_size=640,
                num_electrodes=18,
                dropout=0.5,
                kernel_1=64,
                kernel_2=16,
                F1=8,
                F2=16,
                D=2,
                num_classes=2,
            )
            return builder
        case "CE-stSENet":
            builder = model_builder(
                CE_stSENet,
                inc=18,
                class_num=2,
                si=128,
            )
            return builder
        case "cspnet":
            builder = model_builder(
                CSPNet,
                chunk_size=128 * 5,
                num_electrodes=18,
                num_classes=2,
                dropout=0.5,
                num_filters_t=20,
                filter_size_t=25,
                num_filters_s=2,
                filter_size_s=-1,
                pool_size_1=100,
                pool_stride_1=25,
            )
            return builder
        case "stnet":
            builder = model_builder(
                STNet, chunk_size=128 * 5, grid_size=(9, 9), num_classes=2, dropout=0.2
            )
            return builder
        case "simple-vit":
            if SimpleViT is None:
                raise NotImplementedError(
                    "simple-vit module could not be loaded from simple-vit.py"
                )
            builder = model_builder(
                SimpleViT,
                chunk_size=128 * 5,
                grid_size=(9, 9),
                t_patch_size=32,
                s_patch_size=(3, 3),
                hid_channels=32,
                depth=3,
                heads=4,
                head_channels=8,
                mlp_channels=32,
                num_classes=2,
            )
            return builder
        case "TSception":
            builder = model_builder(
                TSception,
                num_classes=2,
                input_size=(18, 640),
                sampling_rate=256,
                num_T=9,
                num_S=6,
                hidden=128,
                dropout_rate=0.2,
            )
            return builder
        case "FBMSNet":
            builder = model_builder(FBMSNet, nChan=18, nTime=640, nClass=2)
            return builder
        case "LaBraM":
            builder = model_builder(
                LaBraM,
                chunk_size=128 * 5,
                patch_size=80,
                embed_dim=80,
                depth=6,
                num_heads=6,
                mlp_ratio=1,
                qk_norm=partial(nn.LayerNorm, eps=1e-6),
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.1,
                drop_rate=0.1,
                electrodes=channels,
            )
            return builder
        case "RGNN":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_nodes = 18
            edge_index, edge_weight = initialize_edge_weights(num_nodes=num_nodes)
            builder = model_builder(
                RGNN_Model,
                device=device,
                num_nodes=num_nodes,
                edge_weight=edge_weight,
                edge_index=edge_index,
                num_features=5,
                num_hiddens=64,
                num_classes=2,
                num_layers=4,
                dropout=0.1,
                domain_adaptation=False,
            )
            return builder
        case "DGCNN":
            builder = model_builder(
                DGCNN,
                in_channels=5,
                num_electrodes=18,
                num_layers=2,
                hid_channels=32,
                num_classes=2,
            )
            return builder
        case "DGCNN2":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            num_nodes = 18
            edge_index, edge_weight = initialize_edge_weights(num_nodes=num_nodes)
            builder = model_builder(
                DGCNN_Model,
                device=device,
                num_nodes=num_nodes,
                edge_weight=edge_weight,
                edge_index=edge_index,
                num_features=5,
                num_hiddens=32,
                num_classes=2,
                num_layers=2,
                dropout=0.0,
            )
            return builder
        case "Conformer":
            builder = model_builder(
                Conformer,
                num_classes=2, 
                input_dim=18, 
                encoder_dim=32, 
                num_encoder_layers=3
            )
            return builder
        case "TSLANet":
            builder = model_builder(
                TSLANet,
                num_classes=2,         
                chunk_size=640,        
                num_electrodes=18,     
                patch_size=32,         
                emb_dim=128,           
                dropout_rate=0.15,     
                depth=2                
            )
            return builder
        case "LMDA":
            builder = model_builder(
                LMDA,
                num_classes=2,
                chans=18,
                samples=640,
                depth=9,
                kernel=75,
                channel_depth1=24,
                channel_depth2=9,
                ave_depth=1,
                avepool=5
            )
            return builder
        case "EEG_GNN_SSL":
            adj_mx = get_adjacency_matrix(num_channels=18, adj_type='imt')
            builder = model_builder(
                DCRNNModel_classification,
                num_classes=2,
                adj_mx=adj_mx, 
                num_nodes=18,
                num_rnn_layers=2,
                rnn_units=64,
                input_dim=1,
                max_diffusion_step=2,
                dcgru_activation='tanh',
                filter_type='laplacian',
                dropout=0.1,
            )
            return builder
        case "EEGBandClassifier":
            builder = model_builder(
                EEGBandClassifier,
                n_bands = 3,
            )
            return builder
        case _:
            raise NotImplementedError
