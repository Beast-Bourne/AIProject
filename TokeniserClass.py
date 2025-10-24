import re
        
class Tokeniser:
    def __init__(self, vocab):
        self.Tokenise = vocab
        self.Detokenise = {i:s for s,i in vocab.items()}
        
    def TokeniseText(self, text):
        spliter = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        spliter = [item for item in spliter if item.strip()]
        tokens = [self.Tokenise[s] for s in spliter]
        return tokens
    
    def DetokeniseArray(self, intArray):
        string = " ".join([self.Detokenise[i] for i in intArray])
        string = re.sub(r'\s+([,.?!"()\'])', r'\1', string)
        return string