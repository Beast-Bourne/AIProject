import torch
import torch.nn as nn
import SelfAttentionClass as attention

GPT_CONFIG = {
    "vocabSize": 50257,
    "contextLength": 1024,
    "embeddingDim": 768,
    "numHeads": 12,
    "numLayers": 12,
    "qkvBias": False,
}

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenEmbeddings = nn.Embedding(config["vocabSize"], config["embeddingDim"])
        self.positionEmbeddings = nn.Embedding(config["contextLength"], config["embeddingDim"])
        
        self.transformerBlocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["numLayers"])])

        self.finalNorm = LayerNorm(config["embeddingDim"])
        self.outHead = nn.Linear(config["embeddingDim"], config["vocabSize"], bias=False)

    def forward(self, inputToken):
        batchSize, seqLength = inputToken.shape
        tokenEmbeddings = self.tokenEmbeddings(inputToken)
        positionEmbeddings = self.positionEmbeddings(torch.arange(seqLength, device=inputToken.device))
        x = tokenEmbeddings + positionEmbeddings
        x = self.transformerBlocks(x)
        x = self.finalNorm(x)
        logits = self.outHead(x)
        return logits
    
class LayerNorm(nn.Module):
    def __init__(self, embeddingDim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embeddingDim))
        self.shift = nn.Parameter(torch.zeros(embeddingDim))

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        norm = (input-mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift
    
# non-linear activation function used in the feedforward part of the transformer block
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return 0.5 * input * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (input + 0.044715 * torch.pow(input, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(config["embeddingDim"], 4 * config["embeddingDim"]), 
                                    GELU(), nn.Linear(4 * config["embeddingDim"], config["embeddingDim"]))
        
    def forward(self, input):
        return self.layers(input)

class DeepNeuralNetwork(nn.Module):
    def __init__(self, layerSizes, useShortcut):
        super().__init__()
        self.useShortcut = useShortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layerSizes[0], layerSizes[1]), GELU()),
            nn.Sequential(nn.Linear(layerSizes[1], layerSizes[2]), GELU()),
            nn.Sequential(nn.Linear(layerSizes[2], layerSizes[3]), GELU()),
            nn.Sequential(nn.Linear(layerSizes[3], layerSizes[4]), GELU()),
            nn.Sequential(nn.Linear(layerSizes[4], layerSizes[5]), GELU()),
        ])

    def forward(self, input):
        for layer in self.layers:
            output = layer(input)

            if self.useShortcut and input.shape == output.shape:
                input = input + output
            else:
                input = output
        return input
    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = attention.MultiHeadAttentionWrapper(
            config["embeddingDim"], 
            config["embeddingDim"], 
            config["contextLength"], 
            config["numHeads"], 
            config["qkvBias"])
        self.feedForward = FeedForward(config)
        self.norm1 = LayerNorm(config["embeddingDim"])
        self.norm2 = LayerNorm(config["embeddingDim"])
    
    def forward(self, input):
        shortcut = input
        input = self.norm1(input)
        input = self.attention(input)
        input = input + shortcut

        shortcut = input
        input = self.norm2(input)
        input = self.feedForward(input)
        input = input + shortcut

        return input