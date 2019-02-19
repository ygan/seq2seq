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
    # If you don't define init_token and eos_token, you will not get these token when you get training batch data from train_iter
    # Because you define the init_token and eos_token in here, you can get init_token + sentence + eos_token when you create train, val, test from TranslationDataset.splits
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

    # Create batch and make the length of every sentence in one batch become the same
    # If repeat=True, program will forever run in: 'for b, batch in enumerate(train_iter):'
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
#             (train, val, test), batch_size=2, repeat=False, sort=True, sort_within_batch=False)
# DE.vocab.stoi
# for i in range(5):
#     print(DE.vocab.itos[i])

# for i in range(len(EN.vocab)):
#     print(EN.vocab.itos[i])


# for e in range(3):
#     for b, batch in enumerate(train_iter):
#             src, len_src = batch.src
#             trg, len_trg = batch.trg
#             tensorToCsv2D(src,name='src',path='/home/yj/Documents/Python/Github/seq2seq/data2/gan.txt')
#             tensorToCsv2D(len_src,name='len_src',path='/home/yj/Documents/Python/Github/seq2seq/data2/gan.txt')
#             tensorToCsv2D(trg,name='trg',path='/home/yj/Documents/Python/Github/seq2seq/data2/gan.txt')
#             tensorToCsv2D(len_trg,name='len_trg',path='/home/yj/Documents/Python/Github/seq2seq/data2/gan.txt')


# import numpy
# def tensorToCsv2D(tensor,name='defualt',path=None,token=','):

#     def get_variable_name(variable):
#         callers_local_vars = inspect.currentframe().f_back.f_locals.items()
#         return [var_name for var_name, var_val in callers_local_vars if var_val is variable]

#     name = ''.join(get_variable_name(tensor))

#     assert(path is not None)

#     z = tensor.numpy().tolist()
#     if len(numpy.shape(z)) == 2:
#         with open(path,'a') as f:
#             f.write(name)
#             f.write('\r')
#             for i in range(numpy.shape(z)[0]):
#                 for j in range(numpy.shape(z)[1]):
#                     f.write(str(z[i][j]))
#                     f.write(token)
#                 f.write('\r')
#     elif len(numpy.shape(z)) == 1:
#         with open(path,'a') as f:
#             f.write(name)
#             f.write('\r')
#             for i in range(numpy.shape(z)[0]):
#                 f.write(str(z[i]))
#                 f.write(token)
#             f.write('\r')

# tensorToCsv2D(src,name='src',path='/home/yj/Documents/Python/Github/seq2seq/data/gan.txt')
# tensorToCsv2D(len_src,name='len_src',path='/home/yj/Documents/Python/Github/seq2seq/data/gan.txt')
# tensorToCsv2D(trg,name='trg',path='/home/yj/Documents/Python/Github/seq2seq/data/gan.txt')
# tensorToCsv2D(len_trg,name='len_trg',path='/home/yj/Documents/Python/Github/seq2seq/data/gan.txt')

# with open('/home/yj/Documents/Python/Github/seq2seq/data/gan.txt','w') as f:
#     f.write(str(src))
#     f.write(str(len_src))
#     f.write(str(trg))
#     f.write(str(len_trg))
# f
# z = src.numpy().tolist()
# z[0][0]
# len(numpy.shape(z))
# numpy.shape(z)[0]