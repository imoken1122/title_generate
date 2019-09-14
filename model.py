class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout):
        super().__init__()
        
        self.input_dim =input_dim
        self.emb_dim = emb_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim

        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(emb_dim, enc_hidden_dim,bidirectional = True)
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, word_seq):
        #word_seq = [word_len, batch_size]
        embedd = self.dropout(self.embedding(word_seq))
        #embedd = [word_len, batch_size, emb_dim]
        output, h = self.gru(embedd)
        #output = [word_len, batch_size, hidden_dim * n_direction]
        #h = [n_layer * n_direction, batch_size, hidden_dim]
        #h[-2, :, : ] is the last of the forwards RNN 
        #h[-1, :, : ] is the last of the backwards RNN

        through_h = th.tanh(self.fc(th.cat((h[-2,:,:],h[-1,:,:]),dim = 1))) # hidden_dimで結合
        
        #through_h = [batch_size, dec_hidden_dim]
        #output = [seq_len, batch_size, enc_hidden_dim  *2]
        return output, through_h
        
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Parameter(th.rand(dec_hidden_dim)) 
    
    def forward(self, dec_h, enc_output):
        #dec_h = [batch_size, dec_hidden_dim]
        #enc_output = [seq_len, batch_size, enc_hidden_dim  *2]
        batch_size = enc_output.shape[1]
        seq_len = enc_output.shape[0]
        
        dec_h = dec_h.unsqueeze(1).repeat(1,seq_len,1)
        enc_output = enc_output.permute(1,0,2)
        #dec_h = [batch_size, seq_len, dec_hidden_dim]
        #enc_output = [batch_size, seq_len, enc_hidden_dim  *2]
        
        E = th.tanh(self.attn(th.cat((dec_h, enc_output),dim = 2)))
        #E = [batch_size, seq_len, dec_hidden_dim]
        E = E.permute(0,2,1)
        v = self.v.repeat(batch_size, 1 ).unsqueeze(1)
        #v = [batch_size, 1 , dec_hidden_dim]
        attention = th.bmm(v, E).squeeze(1)
        #attention = [batch_size, seq_len]
        
        return F.softmax(attention, dim = 1)
        
        
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hidden_dim, dec_hidden_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.ouptut_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU((enc_hidden_dim * 2) + emb_dim, dec_hidden_dim)
        self.out = nn.Linear((enc_hidden_dim * 2) + emb_dim+dec_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, word, pre_h, enc_output):
        #word = [batch_size] -> [1, batch_size]
        word = word.unsqueeze(0)
        embedd = self.dropout(self.embedding(word))
        # embedd = [1, batch_size, emb_dim]
        
        a = self.attention(pre_h,enc_output) #a = [batch_size, seq_len]]
        a = a.unsqueeze(1) 
        # a = [batch_size, 1, seq_len]
        enc_output = enc_output.permute(1,0,2) 
        #enc_output = [batch_size, seq_len , enc_hidden_dim*2]
        weight = th.bmm(a,enc_output)
        #weight = [batch_size, 1, enc_hidden_dim * 2]
        weight = weight.permute(1,0,2)
        gru_input = th.cat((embedd, weight), dim = 2)
        #gru_input = [1 , batch_size, enc_hidden_dim*2 + emb_dim]
        output, h = self.gru(gru_input, pre_h.unsqueeze(0))
        
        #output = [seq_len = 1, batch_size, hidden_dim * n_direction]
        #h = [n_layer * n_direction =1, batch_size, hidden_dim]
        assert (output == h).all()
        embedd = embedd.squeeze(0)
        output = output.squeeze(0)
        weight=weight.squeeze(0)
        output_ = self.out(th.cat((output, weight, embedd), dim = 1))
        #ouptut_ = [batch_size, vocab_size]
        return output_, h.squeeze(0)
    
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, article, title, teacher_rato = 0.5):
        #article = [seq_len, batch_size]
        
        batch_size = article.shape[1]
        max_len = title.shape[0]
        title_vocab_size= self.decoder.output_dim

        all_output = th.zeros(max_len, batch_size, title_vocab_size).to(self.device)
        enc_output, pre_h = self.encoder(article)
        output = title[0,:]
        for t in range(1,max_len):
            dec_output, h = self.decoder(output, pre_h, enc_output)
            all_output[t] = dec_output
            flag = random.random() < teacher_rato
            most_proba = dec_output.max(1)[1]
            output = title[t] if flag else most_proba
            
        return all_output
