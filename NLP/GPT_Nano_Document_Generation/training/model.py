import math

import torch
from torch import nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * 4),  # expanding the dimension space 4 times
            nn.ReLU(),  # GELU is used
            nn.Linear(in_features=embed_dim * 4, out_features=embed_dim),
            nn.Dropout(p=0.2)
        )    

    def forward(self, x):
        # x shape : (bs, n_ctx, embed_dim)
        out = self.net(x)  # (bs, n_ctx, embed_dim)
        return out


class Head(nn.Module):
    def __init__(self, head_dim, embed_dim, n_ctx, **kwargs) -> None:
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.embed_dim = embed_dim
        self.n_ctx = n_ctx
        # Q, K, V Linear Layers
        self.query = nn.Linear(in_features=self.embed_dim, out_features=self.head_dim, bias=False)
        self.key = nn.Linear(in_features=self.embed_dim, out_features=self.head_dim, bias=False)
        self.value = nn.Linear(in_features=self.embed_dim, out_features=self.head_dim, bias=False)
        
        self.dropout = nn.Dropout(p=0.2)
        # Lower Triangular Matrix for casual attention mask
        self.register_buffer(name="tril", tensor=torch.tril(torch.ones(size=(self.n_ctx, self.n_ctx))))  # block_size?

    def forward(self, x):
        bs, n_ctx, embed_dim = x.shape
        
        # Self Attention = softmax(Q @ K.T / sqrt(head_dim)) @ V
        
        Q = self.query(x)  # (bs, n_ctx, head_dim)
        K = self.key(x)  # (bs, n_ctx, head_dim)
        V = self.key(x)  # (bs, n_ctx, head_dim)

        # Scaled Attention
        attention = Q @ K.transpose(-1, -2) * (1 / math.sqrt(K.shape[-1]))  # (bs, n_ctx, head_dim) @ (bs, head_dim, n_ctx) = (bs, n_ctx, n_ctx) 
        # Note: K.shape[-1]**-0.5 also works for scaling

        # Casual Attention Mask - Masked Attention to avoid the decoder to see future token
        attention = attention.masked_fill(mask=self.tril[:n_ctx, :n_ctx] == 0, value=float("-inf"))  # (bs, n_ctx, n_ctx)

        # Apply Softmax to normalize the scaled, masked affinities
        attention = F.softmax(attention, dim=-1)  # (bs, n_ctx, n_ctx)
        attention = self.dropout(attention)

        # Perform the weighted aggregation of the values
        out = attention @ V  # (bs, n_ctx, n_ctx) @ (bs, n_ctx, head_dim) = (bs, n_ctx, head_dim)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads, n_ctx, *args, **kwargs) -> None:
        """
        Note: torch nn.MultiHeadAttention() can directly implement this block
        -  multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        - attn_output, attn_output_weights = multihead_attn(Q, K, V)  where Q, K, V shape (bs, n_ctx, head_dim)
        - Also, there is torch.nn.functional.scaled_dot_product_attention 
        """
        super().__init__(*args, **kwargs)
        # Note: Here head_dim is single_head_dim
        self.heads = nn.ModuleList([Head(head_dim, embed_dim, n_ctx)
                                    for _ in range(num_heads)])
        # Projection layer
        self.projection = nn.Linear(in_features=head_dim * num_heads, out_features=embed_dim)
        # Dropout
        self.dropout = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        # x shape:  (bs, n_ctx, embed_dim)
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Each h(x) returns (bs, n_ctx, head_dim) => (bs, n_ctx, num_heads*head_dim)
        out = self.dropout(self.projection(out))  # (bs, n_ctx, embed_dim)
        return out
    
class Blocks(nn.Module):
    def __init__(self, embed_dim, num_heads, n_ctx) -> None:
        super().__init__()
        single_h_dim = embed_dim // num_heads
        # Multi-Head Self Attention
        self.sa = MultiHeadAttention(embed_dim, single_h_dim, num_heads, n_ctx)
        # Feed Forward Networks
        self.ffn = MLP(embed_dim)
        # Layer Norm 1
        self.ln1 = nn.LayerNorm(normalized_shape=embed_dim) # normalize across rows instead of columns as in BN
        # Layer Norm 2
        self.ln2 = nn.LayerNorm(normalized_shape=embed_dim)
    
    def forward(self, x):
        # x shape:  (bs, n_ctx, embed_dim)
        x = x + self.sa(self.ln1(x))  # Add i.e residual/skip connection + multi headed selfattn(layer normed inputs)
        x = x + self.ffn(self.ln2(x))  # residual/skip connection + ffn(layer normed inputs)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, *args) -> None:
        super().__init__()
        # Set the Vars
        self.n_ctx = args.n_ctx  # Context Length i.e num of chars in the sequence = Sequence Length
        self.embed_dim = args.embed_dim  # Embedding dimension of each context in the sequence
        self.vocab_size = vocab_size  # Total number of unique vocab: 65
        self.num_heads = args.num_heads  # Num of heads in Multi-Head Attention
        self.n_layers = args.num_layers  # Num of blocks to run (1 block = LN + MSA + Res + LN + FFN + Res )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Token Embedding Table for all the vocab
        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        # Positional Encoding
        self.position_embedding = nn.Embedding(num_embeddings=self.n_ctx, embedding_dim=self.embed_dim)
        # Blocks with Multi-Head Attn, FFD, LN
        self.blocks = nn.Sequential(*[Blocks(embed_dim=embed_dim, num_heads=num_heads, n_ctx=n_ctx) for _ in self.n_layers])
        # Final Layer Norm before linear head
        self.ln_f = nn.LayerNorm(normalized_shape=self.embed_dim) 
        # Linear Head to conver the transformer last hidden state to output logits correspongin to each vocab
        self.linear_head = nn.Linear(in_features=self.embed_dim, out_features=self.vocab_size)
        
        # better init the layers
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """
        Initialize the weights of Linear and Embedding Layers from normal distribution.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, X, Y=None):
        bs, n_ctx = X.shape  # Y shape is also bs, n_ctx
        token_emb = self.token_embedding(X)  # (bs, n_ctx, embed_dim)
        positional_emb = self.position_embedding(X)  # pos_emb((bs, n_ctx)) = (bs, n_ctx, embed_dim)
        x = token_emb + positional_emb  # (bs, n_ctx, embed_dim) + (bs, n_ctx, embed_dim)  = (bs, n_ctx, embed_dim) # shared for all the layers.
        x = self.blocks(x)  # (bs, n_ctx, embed_dim)
        x = self.ln_f(x)  # (bs, n_ctx, embed_dim)
        logits = self.linear_head(x)  # (bs, n_ctx, vocab_size)

        if Y is None:
            loss = None
        else:
            bs, n_ctx, vocab_size = logits.shape
            logits = logits.view(bs * n_ctx, vocab_size)
            Y = Y.view(bs * n_ctx)
            loss = F.cross_entropy(logits, Y)
        return logits, loss
    
    def generate_text(self, input_ctx, max_num_tokens=1000):
        # input is (bs, n_ctx) array of indices in the current context, n_ctx is the sequence length
        for _ in range(max_num_tokens):
            # Crop input to the last n_ctx tokens 
            input_cond = input_ctx[:, :self.n_ctx]  # (bs, n_ctx)
            # Get the logits
            logits, loss = self(input_cond)  # Goes to the forward function # logits (bs, n_ctx, vocab_size) # Here we are sending the whole prev context everytime and only take the last context logits to further to output next token.
            # Get only the last context
            logits = logits[:, -1, :]  # (bs, vocab_size)
            # Apply softmax to the logits to get probs
            probs = F.softmax(logits, dim=-1)  # (bs, vocab_size)
            # Sample from the distribution
            input_next = torch.multinomial(input=probs, num_samples=1)  # (bs, 1)
            # Append sampled index to the running sequence
            input_ctx = torch.cat((input_ctx, input_next), dim=1)  # (bs, n_ctx+1)
        return input_ctx
