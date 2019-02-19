import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        self.embed = nn.Embedding(input_size, embed_size)

        #input_size=embed_size
        self.gru = nn.GRU(embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    #src=input
    def forward(self, src, hidden=None):
        #src shape: (length of sentence, batch)
        #embedded shape: (length of sentence, batch, embed_size)
        embedded = self.embed(src)

        # **outputs** of shape `(seq_len, batch, num_directions * hidden_size)` = ?*32*1024
        # so outputs[0,0,0] is the out from first word of the first sentence in right direction gru
        # so outputs[0,0,hidden_size] is the out from first word of the first sentence in left direction gru
        # However, you should know it will create outputs[seq_len-1,0,hidden_size] and then outputs[seq_len-2,0,hidden_size] ... 
        # and finally create outputs[0,0,hidden_size] in left direction gru
        outputs, hidden = self.gru(embedded, hidden)    #the shape of hidden is (num_layers * num_directions, batch, hidden_size)
        #You can consider hidden as the final state of gru and equal to final outputs
        
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        # so a sentence was embedded to a matrix of seq_len*512
        # the dimention of every word vector is 512
        return outputs, hidden


class Attention(nn.Module):
    #The parameter you need to calc in gradient is :(1) transformation in nn.Linear; (2)self.v
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size) #linear transformation; math:`y = xA^T + b`
        self.v = nn.Parameter(torch.rand(hidden_size))# you can use torch.zeros instead of torch.rand since you will replace the data later on 'self.v.data.uniform_(-stdv, stdv)'
        stdv = 1. / math.sqrt(self.v.size(0))         # 1/sqrt(512)=0.044194173824159216 
        self.v.data.uniform_(-stdv, stdv)             # uniform distribution
        
    # Calc the attn weight('a hat') showed in my note image that show 'a hat' come from hidden z and encoder_output h
    # But the process of image is more simple which is : 
    # match_function(encoder_output, hidden_z) -> a -> softmax to a hat
    # Now there are more steps: 
    # match_function(encoder_output, hidden_z) -> a -> softmax to energy -> relu(v*energy) -> a hat
    # Here add two more steps which is relu(v*energy)
    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)  # seq_len = the number of calc attention

        # the original shape of hidden is (batch, hidden)
        # hidden.repeat(timestep, 1, 1) means repeat 'timestep' times on the first dimention
        # After that, shape of hidden become (timestep, batch, hidden). And hidden[i] euqal to hidden[j]
        # transpose(0, 1) change the matrix. After transpose, the shape become: (batch, timestep, hidden)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # shape of h: (batch, timestep, hidden)

        encoder_outputs = encoder_outputs.transpose(0, 1)  #  shape of encoder_outputs: (batch, timestep, hidden)
        
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)          #F.relu(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # cat function: hidden [B*T*H] + encoder_outputs [B*T*H] -> [B*T*2H]
        # And then self.attn transform [B*T*2H] -> [B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*T*H] -> [B*H*T]
        # Sometimes you can consider that the energy just come from encoder_output
        # Now energy come from both decoder hidden and encoder_output

        # encoder_outputs.size(0) = batch size
        # original shape of v is [hidden_size], So [H] (copy bathc_size times)-> [B*H] -> [B*1*H]
        # This is the Parameter that need to be calc gradient  
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        # You can consider v as the attn weight that decide which part of energy will be added to decodeer


        # torch.bmm(v, energy) means v*energy. v:[B*1*H]* energy:[B*H*T]
        # It means B times: matrix[1*H] multiply matrix[H*T]
        # matrix[1*H] multiply matrix[H*T] = a matrix[1*T]
        # So finally, the shape of energy is [B*1*T]
        energy = torch.bmm(v, energy)  # [B*1*T]

        return energy.squeeze(1)  # squeeze(1) make: [B*1*T] -> [B*T]


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(output_size, embed_size)#?????
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size) #depend on the encoder hidden size
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    #input is the output of decoder, the first input is symbol of '<sos>'
    #the first last_hidden is come form encoder. And then come from decoder
    #encoder_outputs is come form encoder
    def forward(self, input, last_hidden, encoder_outputs):

        # Get the embedding of the current input word (last output word)
        #unsqueeze(0) will add a one dimention in first index
        # the shape of self.embed(input) output is (batch, embed_size)
        embedded = self.embed(input).unsqueeze(0)  # now the shape of embedded is (1, batch, embed_size)
        embedded = self.dropout(embedded)          # self.embed is a one layer neural network, let the network with dropout

        # Calculate attention weights and apply to encoder outputs
        # The first last_hidden is come from encoder.  And it's shape is (num_layers, batch, hidden_size).
        # So the first last_hidden[-1] is the last hidden state from the last encoder layer
        # After that, the last_hidden come form decoder
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,hidden_size)
        context = context.transpose(0, 1)  # (1,B,hidden_size)

        # Combine embedded input word and attended context, run through RNN
        # embedded is (1, batch, embed_size); context is (1, batch, hidden_size)
        # So rnn_input is (1, batch, embed_size+hidden_size)
        rnn_input = torch.cat([embedded, context], 2) 
        
        output, hidden = self.gru(rnn_input, last_hidden) # shape of output and hidden are as same as (1,batch,hidden_size). I think the data of them are same too.
        
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))# leaner transform:
        output = F.log_softmax(output, dim=1) #shape of output: (batch, en_size)

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)  #Be carefull, the batch_size is not a constant.
        max_len = trg.size(0)     #trg sentence length
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        # original shape of hidden is (num_layers * num_directions, batch, hidden_size)
        #From now on: the hidden is the decoder's hidden!!!
        hidden = hidden[:self.decoder.n_layers] #hidden now is only one final direction hidden state
        

        # output is the first world of a sentence in a batch of trg, since the shape of trg is (seq_len, batch)
        # Actually, trg.data[0, :] is always the same. It should be the symbol of '<sos>' that means the 'start of sentence'.
        # So we can manually create the first output instead of geting from trg
        output = Variable(trg.data[0, :])  #'<sos>' # trg.data[0, :] = trg[0, :], but use Variable function, you need to use trg.data

        # loop max_len-1 time, since the first output is '<sos>'
        # outputs[0] do not contain the output. So you can not use outputs[0] to calc loss. 
        # The outputs[0] should be the same as trg.data[0], so omitting it make sense.
        for t in range(1, max_len):

            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)

            # output shape is (batch, en_size); en_size is the output translation language size
            outputs[t] = output 

            #teacher_forcing_ratio must be 0 when evaluation
            is_teacher = random.random() < teacher_forcing_ratio
            
            # transform output from (batch, en_size) to (batch)
            # since the output is the probility of every world but we just want one world as the real output
            # and this one word should be the same as trg data type for next round rnn input
            # So top1 is (batch) and top1[0] is a number that you can find a word form vocab using this number
            # top1 is the index of max value in the column vector(dim=1) of output.data
            top1 = output.data.max(1)[1] # same data type as trg.data
            
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

