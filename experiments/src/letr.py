import torch
import torch.nn as nn
import torch.nn.functional as F
# (plotting helper removed for the paper experiments)


class ConvolutionalTokenization(nn.Module):
    def __init__(self, input_channels, embedding_dim, num_blocks=1):
        super(ConvolutionalTokenization, self).__init__()
        layers = []
        current_channels = input_channels  # Should be equal to the number of mfcc coefficients
        for i in range(num_blocks):
            out_channels = embedding_dim if i == num_blocks - 1 else min(embedding_dim, 3 * (i + 1))
            # conv2d shape requirements : (batch_size, channels, height, width)
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)))
            layers.append(nn.ReLU())
            
            # layers.append(nn.Dropout2d(p=dropout_prob))  # Dropout added after ReLU

            layers.append(nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
            current_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        # Expecting input shape (batch_size, time_steps, n_input)
        if len(x.shape) == 3: 
            x = x.permute(0, 2, 1).unsqueeze(-1)  
        # elif x.shape[1] != 40: 
        #     raise ValueError(f"Input has {x.shape[1]} channels; expected 40.")
        # print(f'Shape at ConvolutionalTokenization entry : {x.shape} ')
        return self.conv_block(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, max_length=None):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        # Using reduced MLP with an expansion factor of 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # Expansion by 2, as per the reduced MLP design
            nn.GELU(),

            # nn.Dropout(p=dropout_prob),  # Dropout added in the MLP
            
            nn.Linear(embed_dim * 2, embed_dim)  # Reduce back to original
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.relative_position_encoding = RelativePositionEncoding(max_length, embed_dim // num_heads) if max_length else None

    def forward(self, x):
        q, k, v = x, x, x  # Self-attention
        attn_bias = self.relative_position_encoding(q, k) if self.relative_position_encoding else None
        attn_output, _ = self.attention(q, k, v, attn_mask=attn_bias)
        x = self.layer_norm1(x + attn_output)
        mlp_output = self.mlp(x)
        return self.layer_norm2(x + mlp_output)


class SubSampledAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, pooling_size=2):
        super(SubSampledAttentionBlock, self).__init__()
        self.pooling = nn.AvgPool1d(pooling_size)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            # nn.Dropout(p=dropout_prob),  # Dropout added in the MLP
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Pooling step to reduce sequence length
        pooled_x = self.pooling(x.permute(1, 2, 0)).permute(2, 0, 1)  # Apply average pooling
        attn_output, _ = self.attention(pooled_x, pooled_x, pooled_x)  # Self-attention on pooled input
        x = self.layer_norm1(pooled_x + attn_output)  # Layer normalization after attention
        mlp_output = self.mlp(x)  # Pass through MLP
        return self.layer_norm2(x + mlp_output)  # Final output after adding the residual


class RelativePositionEncoding(nn.Module):
    def __init__(self, max_length, embed_dim):
        super(RelativePositionEncoding, self).__init__()
        self.relative_bias = nn.Parameter(torch.randn(max_length * 2 - 1, embed_dim))
        self.max_length = max_length

    def forward(self, q, k):
        # q, k: (sequence_length, batch_size, embed_dim)
        seq_len = q.size(0)
        relative_indices = torch.arange(-seq_len + 1, seq_len, device=q.device)
        relative_indices += (self.max_length - 1)  # Shift index to be positive
        relative_bias = self.relative_bias[relative_indices]  # Get relative biases
        return relative_bias


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout=dropout) if i % 2 == 0 
            else SubSampledAttentionBlock(embed_dim, num_heads)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SequencePooling(nn.Module):
    def __init__(self, embed_dim):
        super(SequencePooling, self).__init__()
        self.fc = nn.Linear(embed_dim, 1)  # Linear layer to project each embedding to a single value

    def forward(self, x):
        # x shape: (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, embed_dim)
        weights = F.softmax(self.fc(x), dim=1)  # Softmax across the sequence length (dim=1)
        pooled_output = torch.sum(weights * x, dim=1)  # Weighted sum of the sequence
        return pooled_output  # Output shape: (batch_size, embed_dim)


class LeTR(nn.Module):
    def __init__(self, input_channels, num_classes, embed_dim, num_heads, num_layers, max_length, dropout):
        super(LeTR, self).__init__()
        self.max_length = max_length # Try : 400 for raw data, 220 for noted data. 110 for google speech commands dataset
        self.tokenizer = ConvolutionalTokenization(input_channels, embed_dim)
        self.transformer = TransformerEncoder(num_layers, embed_dim, num_heads, dropout=dropout)
        self.sequence_pooling = SequencePooling(embed_dim)

        # self.dropout = nn.Dropout(p=dropout_prob)

        self.classifier_head = nn.Linear(embed_dim, num_classes)
        self.distillation_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if len(x.shape) > 3 : 
            x = x.squeeze(1)
        _, time_steps, _ = x.shape
        # print(f"At Transformer Forward : ", x.shape)
        # entry_length = self.config_model.get('max_length')
        if time_steps < self.max_length:
            padding_amount = self.max_length - time_steps
            # Pad the last 2 dimensions (https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html)
            x = F.pad(x, (0, 0, 0, padding_amount))  # (padding_left, padding_right, padding_top, padding_bottom)
        elif time_steps > self.max_length:
            x = x[:, :self.max_length, :]

        # plot_mfcc_in_LeTR(x)

        x = self.tokenizer(x)  # Shape: (batch_size, embed_dim, T//2, F//2)
        # print(f"After Tokenization {x.shape}")
        x = x.flatten(2).permute(2, 0, 1)  # Reshape to (T//2 * F//2, batch_size, embed_dim)
        # print(f"After reshaping {x.shape}")
        x = self.transformer(x)  # Shape: (sequence_length, batch_size, embed_dim)
        # print(f"After Transformer {x.shape}")
        pooled_output = self.sequence_pooling(x)  # Shape: (batch_size, embed_dim)
        # print(f"After pooling {pooled_output.shape}")
        class_logits = self.classifier_head(pooled_output)  # Shape: (batch_size, num_classes)
        # print(f"After Classifier Head {class_logits.shape}")
        distill_logits = self.distillation_head(pooled_output)  # Shape: (batch_size, num_classes)
        # print(f"After distillation_head {distill_logits.shape}")
        return class_logits, distill_logits