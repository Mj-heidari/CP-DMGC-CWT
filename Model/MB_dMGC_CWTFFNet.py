import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleTemporalConv(nn.Module):
    """
    Multi-Scale Temporal-Conv Branch (MBdMGC-CWTFFNet)
    --------------------------------------------------
    Inputs:
        x: Tensor of shape (B, C, S)  # B=batch, C=channels, S=time samples

    Behavior:
        - Applies n parallel depthwise temporal Conv1d blocks:
            Conv1d(in=C, out=C, groups=C, kernel_size[k], stride[k], padding)
            + BatchNorm1d(C) + ELU
        - Concatenates outputs along time dimension to form:
            FT: (B, C, sum_k T_k)

    Args:
        in_channels (int): number of EEG channels C
        kernel_sizes (list[int]): kernel sizes per branch (length n)
        strides (int | list[int], optional): stride(s) per branch. Default 1.
        paddings ('same' | int | list[int], optional): padding(s) per branch.
            - 'same' keeps time length when stride=1 (PyTorch≥1.10 supports padding='same').
            - Or provide int/list to control T_k explicitly.
        dilations (int | list[int], optional): dilation(s) per branch. Default 1.
        use_bias (bool): whether Conv1d uses bias. Default False (BN present).
    """
    def __init__(
        self,
        in_channels: int,
        kernel_sizes,
        strides=1,
        paddings='same',
        dilations=1,
        use_bias=False,
    ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        n = len(kernel_sizes)

        # Normalize list-like hyperparams to per-branch lists
        def to_list(x):
            return x if isinstance(x, (list, tuple)) else [x] * n

        strides   = to_list(strides)
        dilations = to_list(dilations)
        paddings  = paddings if isinstance(paddings, (list, tuple)) else [paddings] * n

        self.in_channels = in_channels
        self.n_branches = n
        self.branches = nn.ModuleList()

        for k in range(n):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,   # depthwise -> preserve C
                kernel_size=kernel_sizes[k],
                stride=strides[k],
                padding=paddings[k],
                dilation=dilations[k],
                groups=in_channels,         # depthwise over channels
                bias=use_bias,
            )
            bn = nn.BatchNorm1d(in_channels)
            act = nn.ELU(inplace=True)
            self.branches.append(nn.Sequential(conv, bn, act))

        self._init_weights()

    def _init_weights(self):
        # Kaiming init for ELU
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, S)
        returns FT: (B, C, sum_k T_k)  # concatenated along time
        """
        outs = [branch(x) for branch in self.branches]
        # Concatenate along time dimension (dim=2)
        FT = torch.cat(outs, dim=2)
        return FT

class MultiBandSpectralConv(nn.Module):
    """
    Multi-Band Spectral-Conv Branch using fixed Db4 WaveConv operators.

    - Performs L-level DWT (L = floor(log2(fs)) - 3, min 1).
    - At each level: convolution with lowpass (h) and highpass (g) Db4 filters
      (filters are fixed / non-learnable), then decimation by 2.
    - Gathers detail coefficients at each level and the final approximation.
    - Maps levels to physiological bands (delta, theta, alpha, beta, gamma)
      by finding the decomposition level whose frequency range overlaps the band.
    - Returns F_delta..F_gamma (each shape (B, C, H_l)) and FR concatenated along time dim.
    """

    def __init__(self, fs: int):
        super().__init__()
        self.fs = fs
        L = max(1, math.floor(math.log2(fs)) - 3)
        self.levels = L

        # db4 scaling (lowpass) coefficients (orthonormal)
        # (values commonly used for Daubechies 4)
        h = [
            -0.010597401785069032,
             0.032883011666982945,
             0.030841381835560763,
            -0.18703481171888114,
            -0.02798376941698385,
             0.6308807679295904,
             0.7148465705529154,
             0.2303778133088964
        ]
        h = torch.tensor(h, dtype=torch.float32)  # lowpass
        # highpass filter for decomposition: g[k] = (-1)^{k} * h[::-1][k]
        g = torch.tensor([( (-1)**k ) * v for k, v in enumerate(h.flip(0))], dtype=torch.float32)

        # Save filters as buffers (non-learnable)
        self.register_buffer('db4_low', h)   # shape (K,)
        self.register_buffer('db4_high', g)  # shape (K,)

    def _conv_groups(self, x, filt):
        # x: (B, C, S)
        # filt: 1D tensor (K,)
        B, C, S = x.shape
        K = filt.shape[0]
        # Prepare weight shape (C, 1, K) to perform grouped conv (depthwise per channel)
        w = filt.view(1, 1, K).repeat(C, 1, 1)  # (C,1,K)
        # pad signal to mimic symmetric extension (DWT usually uses signal extension)
        # we'll use reflection padding of size K-1 on the left to keep alignment
        pad = K - 1
        x_p = F.pad(x, (pad, 0), mode='reflect')  # pad left
        # grouped conv: in_channels=C, out_channels=C, groups=C
        out = F.conv1d(x_p, w, bias=None, stride=1, groups=C)
        return out  # shape (B, C, S + pad - (K-1)) => (B, C, S)

    def forward(self, x):
        """
        x: (B, C, S)
        returns:
            F_delta, F_theta, F_alpha, F_beta, F_gamma  (each (B, C, H_i))
            FR: (B, C, sum(H_i)) concatenated along time dim
        """
        B, C, S = x.shape
        approx = x  # initial approximation coefficients
        details = []  # detail coefficients at each level 1..L
        approximations = []  # approximation after each level (we keep last too)

        # iterative DWT decomposition
        for level in range(1, self.levels + 1):
            # lowpass conv then downsample (approximation)
            low = self._conv_groups(approx, self.db4_low)     # (B, C, S_current)
            high = self._conv_groups(approx, self.db4_high)   # (B, C, S_current)

            # downsample by 2 (decimation)
            low_ds = low[..., ::2]   # approximation for next level
            high_ds = high[..., ::2] # detail coefficients at this level

            details.append(high_ds)       # detail for level l : freq band (fs/2^{l+1}, fs/2^{l})
            approximations.append(low_ds) # store approximations too
            approx = low_ds                # feed into next iteration

        # after L levels: 'approx' is the final approximation (lowest freq)
        # Define physiological bands (Hz)
        bands = {
            'delta': (0.0, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta' : (13.0, 30.0),
            'gamma': (30.0, 50.0)
        }

        # compute frequency band for each decomposition output:
        # detail level l corresponds to freq (fs / 2^{l+1}, fs/2^{l})
        # final approximation (after L) corresponds to (0, fs/2^{L+1})
        level_bands = []
        for l in range(1, self.levels + 1):
            low_f = self.fs / (2 ** (l + 1))
            high_f = self.fs / (2 ** l)
            level_bands.append( (low_f, high_f) )  # detail l

        approx_band = (0.0, self.fs / (2 ** (self.levels + 1)))

        # Helper: compute overlap between two freq intervals
        def overlap(a, b):
            lo = max(a[0], b[0])
            hi = min(a[1], b[1])
            return max(0.0, hi - lo)

        # Map each physiological band to the decomposition component (detail level or final approx)
        # Choose the component (one of details[level-1] or approx) with maximal overlap.
        band_to_tensor = {}
        for name, band in bands.items():
            best_score = -1.0
            best_tensor = None

            # check details
            for idx, lvl_band in enumerate(level_bands):
                score = overlap(band, lvl_band)
                if score > best_score:
                    best_score = score
                    best_tensor = details[idx]

            # check approximation (delta often maps here strongly)
            score_approx = overlap(band, approx_band)
            if score_approx > best_score:
                best_score = score_approx
                best_tensor = approx

            # If no overlap found (e.g., fs too low), fallback heuristics:
            if best_tensor is None:
                # fallback: choose nearest-level detail by band center
                band_center = 0.5 * (band[0] + band[1])
                best_idx = min(range(len(level_bands)),
                               key=lambda i: abs(0.5*(level_bands[i][0]+level_bands[i][1]) - band_center))
                best_tensor = details[best_idx] if details else approx

            band_to_tensor[name] = best_tensor  # (B, C, H_level)

        # Extract band tensors in order delta, theta, alpha, beta, gamma
        F_delta = band_to_tensor['delta']
        F_theta = band_to_tensor['theta']
        F_alpha = band_to_tensor['alpha']
        F_beta  = band_to_tensor['beta']
        F_gamma = band_to_tensor['gamma']

        # Concatenate along time dimension (dim=2). They may have different lengths.
        FR = torch.cat([F_delta, F_theta, F_alpha, F_beta, F_gamma], dim=2)

        return F_delta, F_theta, F_alpha, F_beta, F_gamma, FR

class MultiChannelSpatialEncoding(nn.Module):
    """
    Multi-Channel Spatial-Encoding Branch
    -------------------------------------
    Inputs:
        x: (B, C, S)  # EEG trial (C channels, S samples)

    Components:
      1) Channel position encoder:
         - Builds adjacency matrix A ∈ R^{C×C} from electrode distances.
         - Formula:
             a_ij = 1/u_ij              if u_ij < mean(U)
                  = 0                   if u_ij ≥ mean(U)
                  = 1 / mean({u_ij<u})  if i == j
      2) Spatial feature encoder:
         - Channel-wise convolution across (C, S) plane.
         - Extracts multi-channel spatial features FS ∈ R^{B×C×DS}.
    """

    def __init__(self, channel_distances, out_channels=32, kernel_size=(3, 5)):
        """
        Args:
            channel_distances (torch.Tensor): (C, C) matrix of Euclidean distances
                between electrodes (from electrode positions).
            out_channels (int): output channels of spatial encoder.
            kernel_size (tuple): kernel size for Conv2d (channels × time).
        """
        super().__init__()
        assert channel_distances.ndim == 2 and channel_distances.shape[0] == channel_distances.shape[1]
        self.C = channel_distances.shape[0]

        # build adjacency matrix A ∈ R^{C×C}
        self.register_buffer("A", self._build_adjacency(channel_distances))

        # spatial feature encoder: Conv2d over (channels, time)
        # input reshaped to (B, 1, C, S)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def _build_adjacency(self, dist: torch.Tensor):
        """
        Build adjacency matrix A from electrode distance matrix.
        """
        C = dist.shape[0]
        mask = torch.ones_like(dist, dtype=torch.bool)
        mask.fill_diagonal_(False)

        U = dist[mask]  # all u_ij, i != j
        mean_U = U.mean()

        A = torch.zeros_like(dist)

        for i in range(C):
            for j in range(C):
                if i == j:
                    # self-loop weight
                    valid_vals = U[U < mean_U]
                    if len(valid_vals) > 0:
                        A[i, j] = 1.0 / valid_vals.mean()
                    else:
                        A[i, j] = 1.0
                else:
                    if dist[i, j] < mean_U:
                        A[i, j] = 1.0 / dist[i, j]
                    else:
                        A[i, j] = 0.0
        return A

    def forward(self, x):
        """
        Args:
            x: (B, C, S)

        Returns:
            FS: (B, C, D_S)  # spatial features
            A:  (C, C) adjacency matrix
        """
        B, C, S = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}"

        # Spatial feature encoding
        x_in = x.unsqueeze(1)          # (B, 1, C, S)
        feat = self.spatial_conv(x_in) # (B, out_ch, C, S)


        feat = feat.transpose(1, 2)        # (B, C, out_ch, S)
        FS = feat.flatten(2,3)        # (B, C, S*out_ch)

        return FS, self.A

class DynamicGraphConv(nn.Module):
    """
    Point-Wise Dynamic Multi-Graph Convolution Network (dMGCN)

    Args:
        C (int): number of EEG channels
        D (int): feature dimension of input map (DT, DS, or DR)
        reduction (int): reduction ratio 'r' for self-gating
    Inputs:
        x: (B, C, D)  feature map (FT, FS, or FR)
        A_init: (C, C) initial adjacency matrix from channel encoder
    Returns:
        G: (B, C, D)  dynamic graph features (GT, GS, or GR)
        A_dyn: (C, C) learned dynamic adjacency matrix
    """

    def __init__(self, C: int, D: int, reduction: int = 16):
        super().__init__()
        self.C = C
        self.D = D
        self.reduction = reduction

        # --- Self-gating for adjacency matrix ---
        self.fc1 = nn.Linear(C * C, (C * C) // reduction, bias=True)
        self.fc2 = nn.Linear((C * C) // reduction, C * C, bias=True)

        # --- Point-wise convolution kernels (two-layer MLP over feature dim) ---
        self.q1 = nn.Linear(D, D, bias=True)
        self.q2 = nn.Linear(D, D, bias=True)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.q1, self.q2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, A_init: torch.Tensor):
        """
        Args:
            x: (B, C, D)
            A_init: (C, C)
        Returns:
            G: (B, C, D)
            A_dyn: (C, C)
        """
        B, C, D = x.shape
        assert C == self.C and D == self.D, f"Expected ({self.C},{self.D}), got ({C},{D})"

        # --- Step 1: Dynamic adjacency via self-gating ---
        A_vec = A_init.view(-1)                          # (C*C,)
        A_mid = F.elu(self.fc1(A_vec))                   # (C*C)//r
        A_dyn = F.relu(self.fc2(A_mid))                  # (C*C,)
        A_dyn = A_dyn.view(C, C)                         # reshape → (C,C)

        # Degree normalization: D^-1 A
        deg = A_dyn.sum(dim=1, keepdim=True) + 1e-6      # (C,1)
        A_norm = A_dyn / deg                             # row-normalized adjacency

        # --- Step 2: Point-wise graph convolution ---
        h = F.elu(self.q1(x))            # (B,C,D)
        h = self.q2(h)                   # (B,C,D)

        # message passing: (C,C) @ (B,C,D)
        # Use einsum: (C,C) * (B,C,D) -> (B,C,D)
        Axh = torch.einsum("ij,bjd->bid", A_norm, h)

        # residual + activation
        G = F.elu(Axh + x)               # (B,C,D)

        return G, A_dyn

class CWTFFNet(nn.Module):
    def __init__(self, in_dims, d_k=64, d_v=64, hidden_dim=128, dropout=0.1, num_classes=2):
        """
        Channel-Weighted Transformer Feature Fusion Network (CWTFFNet).

        Args:
            in_dims (list[int]): Input feature dims for [Temporal, Spatial, Spectral].
            d_k (int): Dimension for queries/keys.
            d_v (int): Dimension for values.
            hidden_dim (int): Hidden size for classifier.
            dropout (float): Dropout probability.
            num_classes (int): Number of output classes.
        """
        super(CWTFFNet, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

        # Local CW-MHSA (3 heads)
        self.WQ = nn.ModuleList([nn.Linear(in_dim, d_k) for in_dim in in_dims])
        self.WK = nn.ModuleList([nn.Linear(in_dim, d_k) for in_dim in in_dims])
        self.WV = nn.ModuleList([nn.Linear(in_dim, d_v) for in_dim in in_dims])

        # Global CW-FFB (keeps feature dim)
        self.ln = nn.LayerNorm(3 * d_v)
        self.fc1 = nn.Linear(3 * d_v, 3 * d_v)
        self.fc2 = nn.Linear(3 * d_v, 3 * d_v)

        # Feedforward module FM
        self.fm = nn.Sequential(
            nn.Linear(3 * d_v, 3 * d_v),
            nn.ELU(),
            nn.Linear(3 * d_v, 3 * d_v),
        )

        self.dropout = nn.Dropout(dropout)

        # Classifier (input is flattened later → set dynamically)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.out_fc1 = None
        self.out_fc2 = None

    def forward(self, G_list, A_list, A_init):
        """
        Forward pass for CWTFFNet.

        Args:
            G_list (list[Tensor]): [GT, GS, GR], each ∈ (B, C, D).
            A_list (list[Tensor]): [AT, AS, AR], each ∈ (B, C, C).
            A_init (Tensor): Initialized adjacency matrix A ∈ (C, C).
        Returns:
            logits (Tensor): (B, num_classes).
            Z_fused (Tensor): (B, C, 3*d_v).
        """
        B, C, _ = G_list[0].shape
        Z_heads = []

        # 1) Local CW-MHSA
        for i in range(3):
            Q = self.WQ[i](G_list[i])  # (B, C, d_k)
            K = self.WK[i](G_list[i])  # (B, C, d_k)
            V = self.WV[i](G_list[i])  # (B, C, d_v)

            attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, C, C)
            attn = F.softmax(attn, dim=-1)

            Z = torch.matmul(A_list[i], torch.matmul(attn, V))  # (B, C, d_v)
            Z_heads.append(Z)

        Z_local = torch.cat(Z_heads, dim=-1)  # (B, C, 3*d_v)

        # 2) Global CW-FFB
        Z_global = torch.matmul(A_init.unsqueeze(0), Z_local)  # (B, C, 3*d_v)
        channel_weights = F.softmax(Z_local.mean(dim=1, keepdim=True), dim=-1)  # (B, 1, 3*d_v)
        Z_global = Z_global * channel_weights  # channel-wise weighting

        # 3) Feedforward module
        Z_ln = self.ln(Z_global)
        Z_fc = self.fc2(F.relu(self.fc1(Z_ln)))
        Z_fused = Z_fc + self.fm(Z_global)  # residual (B, C, 3*d_v)

        # 4) Classifier
        out = Z_fused.reshape(B, -1)  # flatten (B, C*3*d_v)

        # Lazy init classifier
        if self.out_fc1 is None:
            in_dim_total = out.shape[-1]  # C * 3*d_v
            self.out_fc1 = nn.Linear(in_dim_total, self.hidden_dim).to(out.device)
            self.out_fc2 = nn.Linear(self.hidden_dim, self.num_classes).to(out.device)

        out = F.relu(self.out_fc1(out))
        logits = self.out_fc2(out)

        return logits, Z_fused

class MB_dMGC_CWTFFNet(nn.Module):
    def __init__(self, in_ch=18, sampling_rate = 128, d_model=64, d_k=64, d_v=64, coords = None):

      super(MB_dMGC_CWTFFNet, self).__init__()

      DT = 1920
      DS = 5120
      DR = 640

      self.mstc = MultiScaleTemporalConv(
          in_channels=in_ch,
          kernel_sizes=[5, 11, 23],   # different temporal spans
          strides=1,                  # keep time length if paddings='same'
          paddings='same',            # requires PyTorch with padding='same' support
      )

      self.mbsc = MultiBandSpectralConv(fs=sampling_rate)

      if not coords:
        coords = torch.rand(in_ch, 3)  # (x,y,z) positions of electrodes

      self.dist = torch.cdist(coords, coords, p=2)  # Euclidean distance matrix

      self.mcse = MultiChannelSpatialEncoding(channel_distances=self.dist, out_channels=8, kernel_size=(3, 5))


      self.dgc_T = DynamicGraphConv(in_ch, DT, reduction=16)
      self.dgc_S = DynamicGraphConv(in_ch, DS, reduction=16)
      self.dgc_R = DynamicGraphConv(in_ch, DR, reduction=16)

      # CWTFFNet() 

      self.cwtff = CWTFFNet([DT, DS, DR])

    def forward(self,x):

      FT = self.mstc(x)
      Fd, Ft, Fa, Fb, Fg, FR = self.mbsc(x)

      FS, A = self.mcse(x)

    #   print(FT.shape)
    #   print(FS.shape)
    #   print(FR.shape)

      GT, AT = self.dgc_T(FT, A)
      GS, AS = self.dgc_S(FS, A)
      GR, AR = self.dgc_R(FR, A) 
    #   print(GT.shape)
    #   print(GS.shape)
    #   print(GR.shape)

      result = self.cwtff([GT, GS, GR], [AT, AS, AR], A)

      return result[0]


# Example input: batch of EEG trials
B, C, T = 4, 18, 640
x = torch.randn(B, C, T)  # EEG trials
labels = torch.tensor([0, 1, 0, 1])  # binary labels

model = MB_dMGC_CWTFFNet()
result = model(x)
print(result)
