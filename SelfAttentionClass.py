import torch

class SelfAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        queries = self.W_query(inputs)
        keys = self.W_Key(inputs)
        values = self.W_Value(inputs)

        d_k = keys.shape[1]
        queryAttentionScores = queries @ keys.T # the attention scores are based on the dot product of the query and key weights
        queryAttentWeights = torch.softmax(queryAttentionScores / d_k**0.5, dim=-1)

        contextVector = queryAttentWeights @ values
        return contextVector
