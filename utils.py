import re
import spacy
import torch
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k, TranslationDataset


def load_dataset(batch_size):
    spacy_de = spacy.load('de')#run it on your env or virtrual env:#python -m spacy download de
    spacy_en = spacy.load('en')#run it on your env or virtrual env:#python -m spacy download en
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    # create dataset according to Field object.
    # Field define the basic token and tokenize. 
    # Field can create vocab.
    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    
    #you can find: len(val.examples)=1014; len(test.examples)=1000; len(train.examples)=29000 in Multi30k.splits...
    #train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN)) 

    #I download the data and read it directly:
    #if your file name is not the same as defualt, you must change the function input parameter: train='train', validation='val', test='test'
    #exts parameter is the data file ext name.
    #So the data file depends on the parameter:path+(train\validation\test)+exts
    train, val, test = TranslationDataset.splits(path='./data/',exts=('.de', '.en'), fields=(DE, EN)) 
    
    #build vocabury
    #You can find one word from: DE.vocab.itos[0], it will depend on the order of frenquency
    #You can also find the index of word from: DE.vocab.stoi['word name']
    #It will automatically create the '<pad>' into vocab even you never use it. The '<pad>' sometimes only be used after creating iterators. 
    #It is the same to unkonw_token '<pad>'. If you want: init_token='<sos>', eos_token='<eos>', 
    #you need to give a arguement in creating the Field object.
    DE.build_vocab(train.src, min_freq=2)     # you can just use DE.build_vocab(train, min_freq=2), but not: DE.build_vocab(train.trg, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000) # you can just use EN.build_vocab(train, max_size=10000)

    #Create batch and make the length of every sentence in one batch become the same
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN

# import re
# import spacy
# import torch
# from torchtext.data import Field, BucketIterator
# from torchtext.datasets import Multi30k, TranslationDataset


# spacy_de = spacy.load('de')#run it on your env or virtrual env:#python -m spacy download de
# spacy_en = spacy.load('en')#run it on your env or virtrual env:#python -m spacy download en
# url = re.compile('(<url>.*</url>)')

# def tokenize_de(text):
#     return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

# def tokenize_en(text):
#     return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

# DE = Field(tokenize=tokenize_de, include_lengths=True,
#             init_token='<sos>', eos_token='<eos>')
# EN = Field(tokenize=tokenize_en, include_lengths=True,
#             init_token='<sos>', eos_token='<eos>')

# #you can find: len(val.examples)=1014; len(test.examples)=1000; len(train.examples)=29000 in Multi30k.splits...
# #train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN)) 

# #I download the data and read it directly:
# #if your file name is not the same as defualt, you must change the function input parameter: train='train', validation='val', test='test'
# #exts parameter is the data file ext name.
# #So the data file depends on the parameter:path+(train\validation\test)+exts
# train, val, test = TranslationDataset.splits(path='./data2/',exts=('.de', '.en'), fields=(DE, EN)) 

# #build vocabury
# #You can find one word from: DE.vocab.itos[0], it will depend on the order of frenquency
# #You can also find the index of word from: DE.vocab.stoi['word name']

# DE.build_vocab(train, min_freq=2)
# EN.build_vocab(train, max_size=10000)

# for i in range(5):
#     print(DE.vocab.itos[i])

# train_iter, val_iter, test_iter = BucketIterator.splits(
#             (train, val, test), batch_size=2, repeat=False)
# DE.vocab.stoi
# for i in range(5):
#     print(DE.vocab.itos[i])

# for i in range(len(EN.vocab)):
#     print(EN.vocab.itos[i])

# for b, batch in enumerate(train_iter):
#         src, len_src = batch.src
#         trg, len_trg = batch.trg
#         print(src)
#         print(len_src)
#         print(trg)
#         print(len_trg)