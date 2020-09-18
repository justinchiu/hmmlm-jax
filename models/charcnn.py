import torch as th
import torch.nn as nn

class Highway(nn.Module):
    def __init__(
        self,
        dim,
        num_layers,
    ):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.gate_linears = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))

    def forward(self, x):
        for l in range(self.num_layers):
            g = self.linears[l](x).relu()
            t = self.gate_linears[l](x).sigmoid()
            x = t * g + (1 - t) * x
        return x


class CharLinear(nn.Module):
    """
    nn.CharLinear
    Replaces Linear layer (word embeddings) with output of CharCNN
    """
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        V,
        emit_dims,
        num_highway=1,
    ):
        super(CharLinear, self).__init__()
        self.V = V
        self.emit_dims = emit_dims

        max_len = max(len(x) for x in V.itos) + 2

        # char vocab
        self.pad_idx = 0
        self.i2c = ["<cpad>", "<bow>", "<eow>"] + sorted(list(set(x for xs in V.itos for x in xs)))
        self.c2i = {c: i for i, c in enumerate(self.i2c)}

        # create char_buffer
        self.register_buffer(
            "char_buffer",
            self.process_vocab(
                V,
                th.LongTensor(len(V), max_len),
            )
        )

        self.char_emb = nn.Embedding(
            len(self.i2c), emb_dim, padding_idx=self.pad_idx,
        )
        if not emit_dims:
            self.kernels = list(range(1, 8))
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, hidden_dim, k, bias=False)
                for k in self.kernels
            ])
        else:
            self.kernels = list(range(1, len(emit_dims)+1))
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, d, k, bias=False)
                for k, d in zip(self.kernels, emit_dims)
            ])

            self.mlp = nn.Sequential(
                Highway(sum(emit_dims), num_highway),
                nn.Linear(sum(emit_dims), hidden_dim, bias=False),
            ) if num_highway > 0 else (
                nn.Linear(sum(emit_dims), hidden_dim, bias=False)
            )

    def process_vocab(self, V, char_buffer):
        c2i = self.c2i
        char_buffer.fill_(0)
        for i, word in enumerate(V.itos):
            procword = ["<bow>"] + list(word) + ["<eow>"]
            for t, char in enumerate(procword):
                char_buffer[i,t] = c2i[char]
        return char_buffer


    def forward(self, x):
        char_embs = self.char_emb(self.char_buffer)
        conv_input = char_embs.transpose(-1, -2)
        # this should be pretty slow.
        outs = []
        for conv in self.convs:
            conv_out = conv(conv_input)
            outs.append(conv_out.max(-1).values.tanh())
        if not self.emit_dims:
            y = th.stack(outs, -1).sum(-1)
        else:
            y = self.mlp(th.cat(outs, -1))
        return x @ y.t()

    def get_embs(self):
        char_embs = self.char_emb(self.char_buffer)
        conv_input = char_embs.transpose(-1, -2)
        # this should be pretty slow.
        outs = []
        for conv in self.convs:
            conv_out = conv(conv_input)
            outs.append(conv_out.max(-1).values.tanh())
        if not self.emit_dims:
            y = th.stack(outs, -1).sum(-1)
        else:
            y = self.mlp(th.cat(outs, -1))
        return y

class WordCharLinear(nn.Module):
    """
    WordCharLinear
    Concatenates linear layer (word embeddings) with output of CharCNN
    """
    def __init__(
        self,
        emb_dim,
        hidden_dim,
        V,
        emit_dims,
        num_highway=1,
    ):
        super(WordCharLinear, self).__init__()
        self.V = V
        self.emit_dims = emit_dims

        max_len = max(len(x) for x in V.itos) + 2

        # char vocab
        self.pad_idx = 0
        self.i2c = ["<cpad>", "<bow>", "<eow>"] + sorted(list(set(x for xs in V.itos for x in xs)))
        self.c2i = {c: i for i, c in enumerate(self.i2c)}

        # create char_buffer
        self.register_buffer(
            "char_buffer",
            self.process_vocab(
                V,
                th.LongTensor(len(V), max_len),
            )
        )

        self.word_emb = nn.Embedding.from_pretrained(V.vectors, freeze=False)
        word_dim = self.word_emb.embedding_dim

        self.char_emb = nn.Embedding(
            len(self.i2c), emb_dim, padding_idx=self.pad_idx,
        )
        if not emit_dims:
            self.kernels = list(range(1, 8))
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, hidden_dim, k, bias=False)
                for k in self.kernels
            ])
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim + word_dim, hidden_dim),
                nn.Tanh(),
            )
        else:
            self.kernels = list(range(1, len(emit_dims)+1))
            self.convs = nn.ModuleList([
                nn.Conv1d(emb_dim, d, k, bias=False)
                for k, d in zip(self.kernels, emit_dims)
            ])

            self.mlp = nn.Sequential(
                Highway(sum(emit_dims) + word_dim, num_highway),
                nn.Linear(sum(emit_dims), hidden_dim, bias=False),
            ) if num_highway > 0 else (
                nn.Linear(sum(emit_dims) + word_dim, hidden_dim, bias=False)
            )

    def process_vocab(self, V, char_buffer):
        c2i = self.c2i
        char_buffer.fill_(0)
        for i, word in enumerate(V.itos):
            procword = ["<bow>"] + list(word) + ["<eow>"]
            for t, char in enumerate(procword):
                char_buffer[i,t] = c2i[char]
        return char_buffer


    def forward(self, x):
        char_embs = self.char_emb(self.char_buffer)
        conv_input = char_embs.transpose(-1, -2)
        # this should be pretty slow.
        outs = []
        for conv in self.convs:
            conv_out = conv(conv_input)
            outs.append(conv_out.max(-1).values.tanh())
        if not self.emit_dims:
            y = self.mlp(th.cat([
                th.stack(outs, -1).sum(-1),
                self.word_emb.weight
            ], -1))
        else:
            y = self.mlp(th.cat(outs + [self.word_emb.weight], -1))
        return x @ y.t()

    def get_embs(self):
        char_embs = self.char_emb(self.char_buffer)
        conv_input = char_embs.transpose(-1, -2)
        # this should be pretty slow.
        outs = []
        for conv in self.convs:
            conv_out = conv(conv_input)
            outs.append(conv_out.max(-1).values.tanh())
        if not self.emit_dims:
            y = th.stack(outs, -1).sum(-1)
        else:
            y = self.mlp(th.cat(outs, -1))
        return y

