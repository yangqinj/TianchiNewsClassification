"""
@author: Qinjuan Yang
@time: 2020-08-15 15:58
@desc: TextCNN model for text classification.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,
                 embeddings,
                 embed_size,
                 num_classes,
                 config):
        super(Model, self).__init__()
        self._embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self._convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (fs, embed_size)) for fs in config.filter_sizes]
        )
        self._dropout = nn.Dropout(config.dropout)
        self._fc = nn.Linear(config.num_filters * len(config.filter_sizes), num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))  # batch_size * num_filters * feature_map_len * 1
        x = x.squeeze()  # batch_size * num_filters * feature_map_len
        # kernel_size equals to feature map length
        x = F.max_pool1d(x, x.shape[-1])  # batch_size * num_filters * 1
        x = x.squeeze()  # batch_size * num_filters
        return x

    def forward(self, x):
        out = self._embedding(x)  # batch_size * seq_len * embed_size
        out = out.unsqueeze(1)  # batch_size * 1 * seq_len * embed_size  -> NCHW
        # batch_size * (num_filters * len(filter_sizes)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self._convs], 1)
        out = self._dropout(out)
        out = self._fc(out)
        return out
