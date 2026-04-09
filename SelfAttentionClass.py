import torch

class SelfAttention(torch.nn.Module):

    def __init__(self, d_in, d_out, contextLength, qkv_bias=False):
        super().__init__()
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.mask = torch.triu(torch.ones(contextLength, contextLength), diagonal=1)

    def forward(self, input):
        num_tokens = input.shape[1]
        queries = self.W_query(input)
        keys = self.W_Key(input)
        values = self.W_Value(input)

        queryAttentionScores = queries @ keys.transpose(1, 2)
        masked = queryAttentionScores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        maskWeight = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)

        contextVector = maskWeight @ values
        return contextVector

class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out, contextLength, numOfHeads, qkv_bias=False):
        super().__init__()
        
        assert (d_out % numOfHeads) == 0, "Output dimension must be divisible by the number of heads"

        self.d_out = d_out
        self.numHeads = numOfHeads
        self.headDim = d_out // numOfHeads

        self.queryWeights = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.keyWeights = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.valueWeights = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.outProj = torch.nn.Linear(d_out, d_out)
        self.mask = torch.triu(torch.ones(contextLength, contextLength), diagonal=1)

    def forward(self, input):
        batchSize, num_tokens, d_in = input.shape

        keys = self.keyWeights(input)
        queries = self.queryWeights(input)
        values = self.valueWeights(input)

        keys = keys.view(batchSize, num_tokens, self.numHeads, self.headDim).transpose(1, 2)
        values = values.view(batchSize, num_tokens, self.numHeads, self.headDim).transpose(1, 2)
        queries = queries.view(batchSize, num_tokens, self.numHeads, self.headDim).transpose(1, 2)

        attentionScores = queries @ keys.transpose(2, 3)
        maskBool = self.mask.bool()[:num_tokens, :num_tokens]
        attentionScores = attentionScores.masked_fill(maskBool, -torch.inf)

        attentionWeights = torch.softmax(attentionScores / keys.shape[-1]**0.5, dim=-1)
        contextVector = (attentionWeights @ values).transpose(1, 2)
        contextVector = contextVector.contiguous().view(batchSize, num_tokens, self.d_out)
        contextVector = self.outProj(contextVector)

        return contextVector