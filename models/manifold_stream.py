import torch
import torch.nn as nn


class ManifoldTransformerStream(nn.Module):
    def __init__(self, input_dim=9765, embed_dim=256, num_heads=4, num_layers=2, max_len=200):
        super().__init__()

        # 1. 降维层 (关键！把 9765 维降到 256 维)
        # 如果不降维，Transformer 的参数量会爆炸，且显存不够
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)

        # 2. 位置编码 (这里用简单的可学习参数)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. (可选) 这是一个临时的分类头，用于单流测试
        self.temp_classifier = nn.Linear(embed_dim, 3)  # 假设3分类

    def forward(self, x, src_key_padding_mask=None):
        # x shape: [Batch, Seq_Len, 9765]

        # 降维 -> [Batch, Seq_Len, 256]
        x = self.input_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 加位置编码
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]

        # Transformer 处理 -> [Batch, Seq_Len, 256]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Masked mean when padding exists.
        if src_key_padding_mask is not None:
            valid = ~src_key_padding_mask
            valid = valid.unsqueeze(-1).type_as(x)
            x = x * valid
            denom = valid.sum(dim=1).clamp(min=1.0)
            out = x.sum(dim=1) / denom
        else:
            # out shape: [Batch, 256]
            out = x.mean(dim=1)

        return out


if __name__ == "__main__":
    # 模拟数据：Batch=4, Time=20, Dim=9765
    dummy_input = torch.randn(4, 20, 9765)

    # 实例化模型
    model = ManifoldTransformerStream()

    # 尝试前向传播
    try:
        output = model(dummy_input)
        print("✅ 模型测试通过！")
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")  # 应该是 [4, 256] (特征) 或 [4, 3] (分类结果)
    except Exception as e:
        print("❌ 模型报错：")
        print(e)
