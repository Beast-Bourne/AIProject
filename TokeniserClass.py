import re
        
# NOTE: this class is deprecated and is not used in the final version of the project.
# It was used to undertand how to a vocab dataset could be generated from the training data, and how to use that vocab dataset to tokenise and detokenise text.
# The final version of the project uses a pre-generated vocab dataset from the GPT-2 model
class Tokeniser:
    def __init__(self, dataset):
        
        # generates the vocab array from the dataset
        # reads all customer given inputs then splits them into individual words and punctuation
        vocab = []
        for i in range(len(dataset)):
            text = dataset['instruction'][i]
            spliter = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            spliter = [item for item in spliter if item.strip()] # strips the white space characters
            vocab += spliter

        # sorts the vocab array alphabetically then removes all duplicate entries
        vocab = sorted(list(set(vocab)))
        vocab.extend(["<|unknown|>", "<|endOfText|>"])
        vocab = {token:integer for integer, token in enumerate(vocab)}
        
        # pass a string in to Tokenise[] to get a token (a unique integer identifier)
        # pass an integer into Detokenise[] to get its corrosponding string
        # This only works for words and tokens in the 'vocab' dataset
        self.Tokenise = vocab
        self.Detokenise = {i:s for s,i in vocab.items()}
        
    def TokeniseText(self, text):
        spliter = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        spliter = [item for item in spliter if item.strip()]
        spliter = [item if item in self.Tokenise else "<|unknown|>" for item in spliter]
        tokens = [self.Tokenise[s] for s in spliter]
        return tokens
    
    def DetokeniseArray(self, intArray):
        string = " ".join([self.Detokenise[i] for i in intArray])
        string = re.sub(r'\s+([,.?!"()\'])', r'\1', string)
        return string