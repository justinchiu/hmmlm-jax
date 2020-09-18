import torch as th
import torch.nn as nn

from .misc import ResidualLayerOld

class StateEmbedding(nn.Module):
    def __init__(self,
        num_embeddings,
        embedding_dim,
        num_embeddings1 = None,
        num_embeddings2 = None,
    ):
        super(StateEmbedding, self).__init__()
        self.num_embeddigns = num_embeddings
        self.embedding_dim = embedding_dim

        self.factored = num_embeddings1 is not None and num_embeddings2 is not None

        if not self.factored:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            assert num_embeddings == num_embeddings1 * num_embeddings2
            self.dim1 = num_embeddings1
            self.dim2 = num_embeddings2
            self.emb1 = nn.Embedding(num_embeddings1, embedding_dim)
            self.emb2 = nn.Embedding(num_embeddings2, embedding_dim)
            """
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Tanh(),
            )
            """
            self.mlp = ResidualLayerOld(embedding_dim * 2, embedding_dim)

    def forward(self, x=None):
        # x: torch.LongTensor(batch, time) or None, the states
        if not self.factored:
            return self.emb(x) if x is not None else self.emb.weight
        if x is not None:
            # inner dim is emb2
            x1 = self.emb1(x // self.dim2)
            x2 = self.emb2(x % self.dim2)
            y = self.mlp(th.cat([x1, x2], -1))
            return y
        else:
            # construct cross product
            xprod = th.cat([
                self.emb1.weight[:,None].expand(self.dim1, self.dim2, self.embedding_dim),
                self.emb2.weight[None,:].expand(self.dim1, self.dim2, self.embedding_dim),
            ], -1)
            return self.mlp(xprod.view(self.dim1 * self.dim2, 2 * self.embedding_dim))

class StateEmbedding2(nn.Module):
    def __init__(self,
        num_embeddings,
        embedding_dim,
        num_embeddings1 = None,
        num_embeddings2 = None,
    ):
        super(StateEmbedding2, self).__init__()
        self.num_embeddigns = num_embeddings
        self.embedding_dim = embedding_dim

        self.factored = num_embeddings1 is not None and num_embeddings2 is not None

        if not self.factored:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            assert num_embeddings == num_embeddings1 * num_embeddings2
            self.num1 = num_embeddings1
            self.num2 = num_embeddings2
            self.num3 = num_embeddings
            self.dim1 = embedding_dim // 2
            #self.dim2 = embedding_dim // 4
            #self.dim3 = embedding_dim // 4
            self.dim3 = embedding_dim // 2
            # cluster emb
            self.emb1 = nn.Embedding(self.num1, self.dim1)
            # index emb
            #self.emb2 = nn.Embedding(self.num2, self.dim2)
            # state emb
            self.emb3 = nn.Embedding(self.num3, self.dim3)
            """
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Tanh(),
            )
            """
            self.mlp = ResidualLayerOld(embedding_dim, embedding_dim)

    def forward(self, x=None):
        # x: torch.LongTensor(batch, time) or None, the states
        if not self.factored:
            return self.emb(x) if x is not None else self.emb.weight
        if x is not None:
            # inner dim is emb2
            x1 = self.emb1(x // self.num2)
            #x2 = self.emb2(x % self.num2)
            x3 = self.emb3(x)
            #y = self.mlp(th.cat([x1, x2, x3], -1))
            y = self.mlp(th.cat([x1, x3], -1))
            return y
        else:
            # construct cross product
            xprod = th.cat([
                self.emb1.weight[:,None].expand(self.num1, self.num2, self.dim1),
                #self.emb2.weight[None,:].expand(self.num1, self.num2, self.dim2),
                self.emb3.weight.view(self.num1, self.num2, self.dim3)
            ], -1)
            return self.mlp(xprod.view(self.num3, self.embedding_dim))

    def share(self, other):
        assert self.factored == other.factored

        if self.factored:
            self.emb1.weight = other.emb1.weight
            #self.emb2.weight = other.emb2.weight
            self.emb3.weight = other.emb3.weight
        else:
            self.emb.weight = other.emb.weight

