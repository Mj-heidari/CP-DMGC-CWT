import torch
import torch.nn as nn

class EEGDepthAttention(nn.Module):
    """
    Build EEG Depth Attention module.
    :arg
    C: num of channels
    W: num of time samples
    k: learnable kernel size
    """
    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W)) # Average pool across channels
        # Conv over time dimension to learn depth attention weights
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)
        self.softmax = nn.Softmax(dim=-2) # Softmax over the 'depth' dimension (originally channels)

    def forward(self, x):
        """
        :arg x: Input tensor (Batch, Depth, Channels, Samples)
        """
        # Pool across channels, keep time samples
        x_pool = self.adaptive_pool(x) # (B, D, 1, W)
        # Transpose to make 'depth' the channel dimension for Conv2d
        x_transpose = x_pool.transpose(-2, -3) # (B, 1, D, W)
        # Apply convolution over the 'depth' dimension
        y = self.conv(x_transpose) # (B, 1, D, W)
        # Apply softmax to get attention weights across depth
        y = self.softmax(y)
        # Transpose back to original format
        y = y.transpose(-2, -3) # (B, D, 1, W)

        # Apply attention weights: Element-wise multiply with original input, scaled by C
        return y * self.C * x


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    Adapts input shape: Expects (batch_size, num_channels, chunk_size),
    internally unsqueezes to (batch_size, 1, num_channels, chunk_size).
    """
    def __init__(self, chans=18, samples=640, num_classes=2, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                 ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        # Learnable channel weights applied across the depth dimension (h)
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        # Temporal Convolution Block
        self.time_conv = nn.Sequential(
            # Pointwise conv to expand depth
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            # Depthwise conv over time
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False, padding=(0, kernel // 2)), # Added padding
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )

        # Calculate output size after time_conv to initialize EEGDepthAttention
        with torch.no_grad():
            dummy_input = torch.ones((1, 1, chans, samples)) # Input shape model expects internally
            dummy_einsum = torch.einsum('bdcw, hdc->bhcw', dummy_input, self.channel_weight)
            dummy_time = self.time_conv(dummy_einsum)
            N_dummy, C_dummy, H_dummy, W_dummy = dummy_time.size()
            self.depthAttention = EEGDepthAttention(W_dummy, C_dummy, k=7) # Initialize DA based on output W

        # Channel Convolution Block (Spatial Filtering)
        self.chanel_conv = nn.Sequential(
            # Pointwise conv
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            # Depthwise conv over channels
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        # Final Pooling and Dropout
        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65), # High dropout rate used in original code
        )

        # Calculate the size of the flattened features for the classifier
        with torch.no_grad():
            dummy_channel = self.chanel_conv(dummy_time)
            dummy_norm = self.norm(dummy_channel)
            n_out_features = dummy_norm.cpu().data.numpy().shape
            print(f'(LMDA Init) Shape after norm layer for classifier input: {n_out_features}') # Print shape for debugging
            classifier_input_size = n_out_features[-1] * n_out_features[-2] * n_out_features[-3]
            print(f'(LMDA Init) Calculated input features for classifier: {classifier_input_size}')
            if classifier_input_size <= 0:
                 raise ValueError(f"Calculated feature size is {classifier_input_size}, which is invalid. Check kernel sizes, pooling, and input dimensions.")

        self.classifier = nn.Linear(classifier_input_size, num_classes)

        # Weight initialization (standard practice)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: Expected shape from train.py (batch_size, num_channels, chunk_size)
                 Example: (B, 18, 640)
        """
        # ---> Add unsqueeze operation here <---
        # Reshape to (batch_size, 1, num_channels, chunk_size)
        x = x.unsqueeze(1)

        # Apply learnable channel weights across the 'depth' dimension (h)
        # einsum: (B, 1, C, W), (depth, 1, C) -> (B, depth, C, W)
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        # Temporal convolution block
        x_time = self.time_conv(x) # (B, channel_depth1, C, W')

        # Depth Attention mechanism
        x_time = self.depthAttention(x_time) # (B, channel_depth1, C, W')

        # Channel convolution block (Spatial filtering)
        x = self.chanel_conv(x_time) # (B, channel_depth2, 1, W'')

        # Pooling and dropout
        x = self.norm(x) # (B, channel_depth2, 1, W''')

        # Flatten features
        features = torch.flatten(x, 1)

        # Classifier
        cls = self.classifier(features) # (B, num_classes)
        return cls