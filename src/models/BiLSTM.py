"""
@author: Qinjuan Yang
@time: 2020-08-20 22:05
@desc: BiLSTM model for text classification.
"""
from torch import nn


class Model(nn.Module):
    def __init__(self,
                 embeddings,
                 embed_size,
                 num_classes,
                 config):
        super(Model, self).__init__()
        self._embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self._lstm = nn.LSTM(embed_size, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self._fc = nn.Linear(config.hidden_size * 2, num_classes)

    def forward(self, x):
        out = self._embedding(x)  # num_batches * seq_len * embed_size

        # out: num_batches * seq_len * (hidden_size * 2)
        # _[0], hn: num_layers * num_batches * hidden_size
        # _[1], cn: num_layers * num_batches * hidden_size
        out, _ = self._lstm(out)

        out = self._fc(out[:, -1, :])  # use the hidden state of the last timestep
        return out
