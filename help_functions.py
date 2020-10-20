from collections import Counter    

PAD = '___PAD___'
UNKNOWN = '___UNKNOWN___'
BOS = '___BOS___'
EOS = '___EOS___'

class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""
    
    def __init__(self, max_voc_size=None, min_word_freq=None, lower=False):
        # String-to-integer mapping
        self.stoi = None
        # Integer-to-string mapping
        self.itos = None
        # Maximally allowed vocabulary size.
        self.max_voc_size = max_voc_size
        self.min_word_freq = min_word_freq
        self.lower = lower
        self.vectors = None
        
    def build(self, seqs):
        """Builds the vocabulary."""
        
        if self.lower:
            seqs = [ [s.lower() for s in seq] for seq in seqs ]
        
        # Sort all words by frequency
        word_freqs = Counter(w for seq in seqs for w in seq)
        word_freqs = sorted(((f, w) for w, f in word_freqs.items()), reverse=True)

        # Build the integer-to-string mapping. The vocabulary starts with the two dummy symbols,
        # and then all words, sorted by frequency. Optionally, limit the vocabulary size.
        
        if self.max_voc_size:
            self.itos = [ w for _, w in word_freqs[:self.max_voc_size] ]
        elif self.min_word_freq:
            self.itos = [ w for f, w in word_freqs if f>=self.min_word_freq]
        else:
            self.itos = [ w for _, w in word_freqs ]

        # Build the string-to-integer map by just inverting the aforementioned map.
        self.stoi = { w: i for i, w in enumerate(self.itos) }
    
    def build_word_occurence_dict(self, text_list):
        word_occurence_dict = {token: [] for token in self.stoi}
        for text in text_list:
            for token in word_occurence_dict:
                word_occurence_dict[token].append(text.lower().split(" ").count(token))

        return word_occurence_dict


    def __len__(self):
        return len(self.itos)