import torch
from torch import nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PatchEmbedding(nn.Module):

    def __init__(self, embed_size: int, patch_size:int, d_model: int, num_patches: int, dropout: float = 0.2):
        super().__init__()
        self.patch_size = patch_size
        self.linear_projection = nn.Linear(embed_size, d_model)
        self.position_embedding = self.positional_embedding(num_patches+1, d_model) # 65 x 128
        self.cls_token = nn.Parameter(torch.rand(1, d_model, device=torch.device(device))) # 1 x 128
        self.dropout = nn.Dropout(dropout)


    def create_patches(self, inputs:torch.Tensor):
        batch_size = inputs.size(0)
        res = inputs.unfold(2, self.patch_size, self.patch_size)  # bs, 3 x 8 x 32 x 4  
        res = res.unfold(3, self.patch_size, self.patch_size)  # bs, 3 x 8 x 8 x 4 x 4  
        res = res.reshape(batch_size, -1, self.patch_size * self.patch_size * 3)

        return res.to(device)  # bs, -1 x 48 == bs x 64 x 48


    def positional_embedding(self, pos_length:int, d_model:int):
        pos_embedded = torch.ones(pos_length, d_model)
        for pos in range(pos_length):
            for i in range(d_model):
                pos_embedded[pos][i] = np.sin(pos / (10000 ** (i/d_model))) if i % 2 == 0 else np.cos(pos / (10000 ** ((i - 1)/d_model)))
                
        return pos_embedded.to(device)


    def forward(self, x:torch.Tensor):
        patches = self.create_patches(x)
        linear_projection = self.linear_projection(patches)

        cls_token = self.cls_token.repeat(linear_projection.size(0), 1, 1) # batch_size x 1 x d_model
        cls_embedding = torch.concat([cls_token, linear_projection], dim=1)

        position_embedding = self.position_embedding.repeat(linear_projection.size(0), 1, 1) # batch_size x n_patches+1 x d_model
        positional_embedding = cls_embedding + position_embedding
        
        return positional_embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_hidden, num_heads):
        super().__init__()
        
        self.d_model    = d_model
        self.d_hidden   = d_hidden
        self.num_heads  = num_heads
        self.d_key      = d_hidden // num_heads
        
        self.Wq = nn.Linear(self.d_model, self.d_hidden)
        self.Wk = nn.Linear(self.d_model, self.d_hidden)
        self.Wv = nn.Linear(self.d_model, self.d_hidden)
        self.Wo = nn.Linear(self.d_hidden, self.d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size = x.size(0)
        Q = self.Wq(x).view(batch_size, -1, self.num_heads, self.d_key).permute(0, 2, 1, 3)
        K = self.Wk(x).view(batch_size, -1, self.num_heads, self.d_key).permute(0, 2, 1, 3)
        V = self.Wv(x).view(batch_size, -1, self.num_heads, self.d_key).permute(0, 2, 1, 3)
        scaled_dot_prod = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.d_key)
        attention = self.softmax(scaled_dot_prod) # (batch_size, n_heads, Q_length, K_length)
        values = torch.matmul(attention, V)
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_key) # (batch_size, Q_length, d_model)
        output = self.Wo(values)

        return output, attention


class Encoder(nn.Module):

    def __init__(self, mlp_filters:int, d_model:int, d_hidden:int, num_heads:int, dropout=0.2):
        super().__init__()
        
        self.mha  = MultiHeadAttention(d_model, d_hidden, num_heads)
        self.mlp  = nn.Sequential(nn.Linear(d_model, mlp_filters),
                                  nn.GELU(),
                                  nn.Linear(mlp_filters, d_model),
                                 )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        output = x
        norm1 = self.norm(output)
        mha, att   = self.mha(norm1)
#         mha   = self.dropout(mha)
        output1 = mha + x
        norm2 = self.norm(output1)
        mlp   = self.mlp(norm2)
#         mlp   = self.dropout(mlp)
        output2 = mlp + output1
    
        return output2, att
     

class ViT(nn.Module):
    def __init__(self, embed_size:int, patch_size:int, d_model:int, d_hidden:int, num_patches:int,
                 mlp_filters:int, num_heads:int, num_layers:int, num_classes:int):
        super().__init__()
        self.embedding_layer = PatchEmbedding(embed_size, patch_size, d_model, num_patches)
        self.encoder_layers  = nn.ModuleList([Encoder(mlp_filters=mlp_filters, d_model=d_model, d_hidden=d_hidden,
                                                     num_heads=num_heads) for _ in range(num_layers)])
        self.mlp_head        = nn.Sequential(nn.LayerNorm(d_model),
                                             nn.Linear(d_model, num_classes)
                                            )
        
    def forward(self, x):
        x = self.embedding_layer(x)
        
        attention_maps = []
        for idx, layer in enumerate(self.encoder_layers):
            x, attn_weights = layer(x)
            attention_maps.append(attn_weights)
        x = self.mlp_head(x[:, 0, :])
        return x, attention_maps


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    
    t = torch.ones((5, 3, 32, 32)).to(device)
    patch_size = 4
    n_patches = (32 * 32) // (patch_size**2)
    embed_dim = 3 *patch_size**2
    model = ViT(embed_size=embed_dim, patch_size=patch_size, d_model=128, d_hidden=128, num_patches=n_patches, 
                mlp_filters=128, num_heads=8, num_layers=4, num_classes=10).to(device)

    logits, attention_maps = model(t)
    np.shape(logits.to(device))
    att = torch.tensor(attention_maps[3][0][0])
    plt.imshow(att.cpu().numpy())

