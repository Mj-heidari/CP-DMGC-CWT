# Combined code for DCRNN Classification Model
# Adapted from https://github.com/tsy935/eeg-gnn-ssl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import math
import mne # Needed for get_adjacency_matrix
# Ensure this import path is correct relative to your project structure
from dataset.adjacency_matrix import euclidean_dist, inverse_mean_threshold_adjacency


# --- Helper Function to Get Adjacency Matrix ---
# Moved from provider.py
def get_adjacency_matrix(num_channels=18, adj_type='imt'):
    """
    Helper function to compute the adjacency matrix based on standard locations.
    Assumes a fixed set of 18 bipolar channels as defined globally.
    """
    print(f"Computing adjacency matrix (type: {adj_type}) for {num_channels} channels...")
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        pos_dict = montage.get_positions()["ch_pos"]

        # Your specific 18 bipolar channels (ensure this list is accurate)
        bipolar_ch_names = [
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP1-F7", "F7-T7",
            "T7-P7", "P7-O1", "FZ-CZ", "CZ-PZ", "FP2-F4", "F4-C4",
            "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
        ]
        if len(bipolar_ch_names) != num_channels:
             print(f"Warning: Code expects {num_channels} channels, but bipolar_ch_names list has {len(bipolar_ch_names)}. Using list length.")
             num_channels = len(bipolar_ch_names)

        positions = []
        missing_count = 0
        for ch_pair in bipolar_ch_names:
            parts = ch_pair.split('-')
            if len(parts) != 2:
                 print(f"Warning: Invalid channel name format '{ch_pair}'. Skipping.")
                 missing_count += 1
                 positions.append(np.array([0.0, 0.0, 0.0])) # Fallback
                 continue

            ch1, ch2 = parts
            # Try finding positions case-insensitively
            pos1 = pos_dict.get(ch1.upper()) or pos_dict.get(ch1)
            pos2 = pos_dict.get(ch2.upper()) or pos_dict.get(ch2)

            if pos1 is not None and pos2 is not None:
                 positions.append((pos1 + pos2) / 2.0)
            else:
                 print(f"Warning: Position missing for {ch1.upper()} or {ch2.upper()} in '{ch_pair}'. Using origin.")
                 missing_count += 1
                 positions.append(np.array([0.0, 0.0, 0.0])) # Fallback

        if missing_count > 0:
             print(f"Warning: Could not find positions for {missing_count} channel pairs.")

        positions_3d = np.array(positions)
        if len(positions_3d) != num_channels:
             raise ValueError(f"Failed to compute positions for all {num_channels} channels. Got {len(positions_3d)}.")

        dist_matrix = euclidean_dist(positions_3d)

        if adj_type == 'imt':
            print("Using Inverse Mean Threshold (IMT) adjacency.")
            adj_mx = inverse_mean_threshold_adjacency(dist_matrix)
        elif adj_type == 'dist':
            print("Using inverse distance adjacency.")
            # Add epsilon before division, handle diagonal after
            adj_mx = 1. / (dist_matrix + 1e-6)
            np.fill_diagonal(adj_mx, 0) # No self-connection based on distance
        else:
             print(f"Warning: Unknown adj_type '{adj_type}', using identity matrix.")
             adj_mx = np.eye(dist_matrix.shape[0])

        print(f"Adjacency matrix shape: {adj_mx.shape}")
        if not np.any(adj_mx - np.diag(np.diag(adj_mx))): # Check if non-diagonal elements exist
             print("Warning: Computed adjacency matrix is diagonal or zero. Check distances/method. Using identity.")
             adj_mx = np.eye(dist_matrix.shape[0])

        return adj_mx

    except Exception as e:
         print(f"Error getting adjacency matrix: {e}. Returning identity matrix.")
         return np.eye(num_channels) # Fallback to identity

# --- Helper Functions (Graph Calculations from utils.py) ---

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Calculate scaled Laplacian matrix L = (2 / lambda_max * L_norm) - I
    where L_norm = I - D^{-1/2} A D^{-1/2}
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        try:
             lambda_max, _ = linalg.eigsh(L, 1, which='LM')
             lambda_max = lambda_max[0]
        except linalg.ArpackNoConvergence:
             print("Warning: eigsh did not converge, using largest absolute eigenvalue via eigs.")
             # Fallback for non-symmetric or convergence issues (slower)
             try:
                 lambda_max, _ = linalg.eigs(L.astype(np.complex128), 1, which='LM') # Use complex dtype for eigs
                 lambda_max = np.real(lambda_max[0])
             except Exception as e:
                  print(f"Warning: eigs also failed ({e}). Using lambda_max=2 as fallback.")
                  lambda_max = 2.0


    # L is coo matrix
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)

    # Check lambda_max value
    if lambda_max < 1e-6 or not np.isfinite(lambda_max):
         print(f"Warning: Invalid lambda_max ({lambda_max}). Using 2 as fallback.")
         lambda_max = 2.0

    L_scaled = (2.0 / lambda_max * L) - I
    return L_scaled.tocoo() # Return as coo_matrix

def calculate_normalized_laplacian(adj):
    """
    Calculate L = I - D^{-1/2} A D^{-1/2}
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_random_walk_matrix(adj_mx):
    """ Calculate D^{-1} A """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def calculate_reverse_random_walk_matrix(adj_mx):
    """ Calculate D^{-1} A^T """
    return calculate_random_walk_matrix(adj_mx.T)

def build_sparse_matrix(L):
    """ Build PyTorch sparse tensor from SciPy sparse matrix """
    L = L.tocoo()
    shape = L.shape
    i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
    v = torch.FloatTensor(L.data)
    # Use torch.sparse_coo_tensor for modern PyTorch versions
    return torch.sparse_coo_tensor(i, v, torch.Size(shape), requires_grad=False)

def last_relevant_pytorch(output, lengths, batch_first=True):
    """ Get the last relevant output state based on sequence lengths """
    lengths = lengths.cpu().long() # Ensure lengths are long integers on CPU
    # Create index for the last valid time step for each sequence
    idx = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)) # (B, Hidden*Nodes)
    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension) # (B, 1, Hidden*Nodes) if batch_first
    # Ensure index is on the same device as the output tensor
    idx = idx.to(output.device)
    # Gather the last relevant output
    last_output = output.gather(time_dimension, idx).squeeze(time_dimension)
    return last_output


# --- DCGRU Cell Components (from cell.py) ---

class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(torch.FloatTensor(size=(self._input_size * num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    def forward(self, supports, inputs, state, output_size):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x = inputs_and_state
        x0 = x
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step > 0:
            supports_mat = []
            # Pre-convert sparse supports to dense if needed, handle device
            for support_sparse in supports:
                if support_sparse.is_sparse:
                    supports_mat.append(support_sparse.to_dense().to(x0.device))
                else: # Assume already dense
                    supports_mat.append(support_sparse.to(x0.device))

            for support in supports_mat: # Now using dense supports
                x1 = torch.einsum('ij,bjd->bid', support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.einsum('ij,bjd->bid', support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(supports) * self._max_diffusion_step + 1
        x = x.permute(0, 2, 1, 3) # (B, N, num_matrices, D_in+D_hid)
        x = x.reshape(batch_size, self._num_nodes, num_matrices * input_size)
        x = x.reshape(-1, num_matrices * input_size) # (B * N, num_matrices * D)

        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """ Diffusion Convolutional GRU Cell """
    def __init__(self, input_dim, num_units, max_diffusion_step, num_nodes,
                 filter_type="laplacian", nonlinearity='tanh', use_gc_for_ru=True):
        super(DCGRUCell, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        if filter_type == "laplacian": self._num_supports = 1
        elif filter_type in ["random_walk", "dual_random_walk"]: self._num_supports = 2
        else: self._num_supports = 1

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports, input_dim=input_dim, hid_dim=num_units,
            num_nodes=num_nodes, max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2, filter_type=filter_type)
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports, input_dim=input_dim, hid_dim=num_units,
            num_nodes=num_nodes, max_diffusion_step=max_diffusion_step,
            output_dim=num_units, filter_type=filter_type)

    @property
    def output_size(self):
        return self._num_nodes * self._num_units

    def forward(self, supports, inputs, state):
        output_size_gate = 2 * self._num_units
        if self._use_gc_for_ru:
            value = torch.sigmoid(self.dconv_gate(supports, inputs, state, output_size_gate))
        else:
             raise NotImplementedError("_fc not implemented for gates")

        value = torch.reshape(value, (-1, self._num_nodes, output_size_gate))
        r, u = torch.split(value, split_size_or_sections=self._num_units, dim=-1)

        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * state + (1 - u) * c
        output = new_state
        return output, new_state

    def init_hidden(self, batch_size):
        param = next(self.parameters())
        return torch.zeros(batch_size, self._num_nodes * self._num_units, device=param.device)


# --- DCRNN Encoder ---

class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        encoding_cells = list()
        encoding_cells.append(
            DCGRUCell(input_dim=input_dim, num_units=hid_dim,
                      max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                      nonlinearity=dcgru_activation, filter_type=filter_type))
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(input_dim=hid_dim, num_units=hid_dim,
                          max_diffusion_step=max_diffusion_step, num_nodes=num_nodes,
                          nonlinearity=dcgru_activation, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        output_hidden = []

        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                output, hidden_state = self.encoding_cells[i_layer](supports, current_inputs[t, ...], hidden_state)
                output_inner.append(output)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(self._device)

        output_hidden = torch.stack(output_hidden, dim=0).to(self._device)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # Ensure initial state is on the correct device
        return torch.stack(init_states, dim=0) # .to(self._device) handled in cell


# --- DCRNN Model for Classification ---

class DCRNNModel_classification(nn.Module):
    """
    DCRNN model adapted for classification.
    Input shape: (batch, num_channels, seq_len) -> adapted internally
    Output shape: (batch, num_classes)
    """
    def __init__(self, num_classes, num_nodes=18, num_rnn_layers=2, rnn_units=64, input_dim=1,
                 max_diffusion_step=2, dcgru_activation='tanh', filter_type='laplacian',
                 dropout=0.1, adj_mx=None, device=None):
        super(DCRNNModel_classification, self).__init__()

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.filter_type = filter_type
        self.max_diffusion_step = max_diffusion_step

        if adj_mx is None:
            print("Warning: No adjacency matrix provided. Using identity matrix.")
            adj_mx = np.eye(num_nodes)

        # Precompute supports (graph diffusion matrices) and store numpy versions
        self._supports_np = self._prepare_supports_numpy(adj_mx, filter_type)
        # Store sparse tensors, move to device in .to() or forward()
        self._supports = [build_sparse_matrix(sp_mat) for sp_mat in self._supports_np]


        self.encoder = DCRNNEncoder(input_dim=input_dim,
                                      max_diffusion_step=max_diffusion_step,
                                      hid_dim=rnn_units, num_nodes=num_nodes,
                                      num_rnn_layers=num_rnn_layers,
                                      dcgru_activation=dcgru_activation,
                                      filter_type=filter_type,
                                      device=self._device)

        self.fc = nn.Linear(rnn_units, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
             nn.init.zeros_(self.fc.bias)

        # Move initial supports to the default device
        self.to(self._device)


    def _prepare_supports_numpy(self, adj_mx, filter_type):
        """ Precompute support matrices as numpy/scipy arrays first """
        supports_np = []
        print(f"Calculating supports for filter type: {filter_type}")
        if filter_type == "laplacian":
            supports_np.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports_np.append(calculate_random_walk_matrix(adj_mx))
        elif filter_type == "dual_random_walk":
            supports_np.append(calculate_random_walk_matrix(adj_mx))
            supports_np.append(calculate_reverse_random_walk_matrix(adj_mx))
        else:
            print(f"Warning: Unknown filter type '{filter_type}'. Using 'laplacian'.")
            supports_np.append(calculate_scaled_laplacian(adj_mx, lambda_max=None))

        print(f"Calculated {len(supports_np)} support matrices.")
        return supports_np

    # Override the .to() method to ensure supports are moved with the model
    def to(self, *args, **kwargs):
        super_result = super().to(*args, **kwargs) # Call the original .to() method first

        # Determine the target device from arguments
        device = None
        if args and isinstance(args[0], (torch.device, str)):
            device = torch.device(args[0])
        elif kwargs and 'device' in kwargs:
            device = torch.device(kwargs['device'])

        if device:
            # Move precomputed supports to the target device
            try:
                self._supports = [build_sparse_matrix(sp_mat).to(device) for sp_mat in self._supports_np]
                print(f"Moved supports to device: {device}")
                # Update device attribute in encoder as well
                self.encoder._device = device
            except AttributeError as e:
                 # This might happen if _supports_np is not yet initialized during the very first .to() call
                 print(f"Debug: Could not move supports during .to(): {e}")
                 pass # Supports will be moved in the first forward pass if needed
        return super_result


    def forward(self, input_seq):
        """
        Args:
            input_seq: Input sequence, shape (batch, num_channels, seq_len)
        """
        # Ensure supports are on the same device as the input
        current_device = input_seq.device
        # Check if supports exist and if the device matches
        if not hasattr(self, '_supports') or not self._supports or self._supports[0].device != current_device:
             print(f"Moving supports to current device: {current_device}")
             # Rebuild sparse tensors on the correct device if they weren't moved by .to() or device changed
             self._supports = [build_sparse_matrix(sp_mat).to(current_device) for sp_mat in self._supports_np]
             # Update encoder device as well
             self.encoder._device = current_device


        # --- Adapt input shape ---
        if input_seq.dim() == 3: # (B, C, T)
             input_seq = input_seq.permute(0, 2, 1) # (B, T, C)
             input_seq = input_seq.unsqueeze(-1)    # (B, T, C, 1) - Add input_dim=1
        elif input_seq.dim() == 4 and input_seq.shape[1] == 1: # (B, 1, C, T)
             input_seq = input_seq.squeeze(1).permute(0, 2, 1).unsqueeze(-1) # (B, T, C, 1)

        batch_size, seq_len, num_nodes, _ = input_seq.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Input has {num_nodes} nodes, but model configured for {self.num_nodes}")

        seq_lengths = torch.full((batch_size,), seq_len, device=self.encoder._device) # Use encoder's device

        # Transpose for encoder: (seq_len, batch, num_nodes, input_dim)
        input_seq = input_seq.permute(1, 0, 2, 3)

        # Initialize encoder hidden state (now correctly placed on device by init_hidden)
        init_hidden_state = self.encoder.init_hidden(batch_size)

        # Encoder forward pass
        _, final_hidden = self.encoder(input_seq, init_hidden_state, self._supports)

        # Output from last layer: (batch_size, seq_len, N*rnn_units)
        output = final_hidden.permute(1, 0, 2)

        # Extract last relevant output state
        last_out = last_relevant_pytorch(output, seq_lengths, batch_first=True)

        # Reshape: (batch_size, num_nodes, rnn_units)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        # last_out = last_out.to(self.encoder._device) # Should already be on correct device

        # Classifier
        logits = self.fc(self.relu(self.dropout(last_out)))

        # Max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)

        return pool_logits