import torch

class InstructionTextGeneration:

    # Converts text to token IDs using the provided tokeniser
    def TextToTokenIds(self, text, tokeniser):
        encoded = tokeniser.encode(text, allowed_special={"<|endoftext|>"})
        encodedTensor = torch.tensor(encoded).unsqueeze(0)
        return encodedTensor

    # Converts token IDs back to text using the provided tokeniser
    def TokenIdsToText(self, tokenIds, tokeniser):
        flat = tokenIds.squeeze(0).tolist()
        return tokeniser.decode(flat)

    # Generates new tokens based on the input token sequence and the model's predictions, using temperature and top-k sampling for diversity
    def GenerateTokensForContext(self, model, inputTokens, maxNewTokens, contextSize, temperature=0.0, topK=None, eosTokenId=None):
        for _ in range(maxNewTokens):
            inputCondition = inputTokens[:, -contextSize:]

            with torch.no_grad():
                logits = model(inputCondition)

            logits = logits[:, -1, :]

            if topK is not None:
                topLogits, topPos = torch.topk(logits, topK)
                newLogits = torch.where(condition=logits<topLogits[:,-1], input=torch.tensor(float("-inf")), other=logits)
            
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                nextToken = torch.multinomial(probs, num_samples=1)
            else:
                nextToken = torch.argmax(logits, dim=-1, keepdim=True)

            # end of sequence token tells the model when to stop generating text
            if nextToken.item() == eosTokenId:
                break

            inputTokens = torch.cat((inputTokens, nextToken), dim=1)

        return inputTokens
    
    # Generates a sample of text based on the provided model, tokeniser, and starting context, and prints the generated text
    def GenerateAndPrintSample(self, model, tokeniser, instructionText, printText=True):
        model.eval()
        contextSize = model.positionEmbeddings.weight.shape[0]
        encoded = self.TextToTokenIds(instructionText, tokeniser)

        with torch.no_grad():
            generatedIds = self.GenerateTokensForContext(model, encoded, 100, contextSize, temperature=0.8, eosTokenId=50256)
        
        decodedText = self.TokenIdsToText(generatedIds, tokeniser)
        decodedText = decodedText[len(instructionText):].replace("### Response:", "").strip()
        if (printText): print(decodedText)
        
        model.train()
        return decodedText
