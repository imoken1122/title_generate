#!pip install adabound
import random
from tqdm import tqdm_notebook as tqdm
import torch as th
from torch import nn,optim
from torch.autograd import Variable as V
import models
import janome
import torchtext
from torchtext import data
from janome.tokenizer import Tokenizer
import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
import torch.nn.functional as F



t_j = Tokenizer()
def _tokenizer(text):
    return t_j.tokenize(text,wakati = True)

def read_data():

    a = data.Field(tokenize=_tokenizer, init_token="<sos>",eos_token = "<eos>",include_lengths = True)
    t = data.Field(tokenize=_tokenizer, init_token="<sos>",eos_token = "<eos>")
    data_loader= data.TabularDataset.splits(
        path='./', train='data/train_2.tsv', test='data/test_2.tsv', format='tsv',
            fields=[('article', ARTICLE), ('title', TITLE)])
    return a,t,data_loader

ARTICLE, TITLE, train,test = read_data()
ARTICLE.build_vocab(train)
TITLE.build_vocab(train)
art_w2v = dict(ARTICLE.vocab.stoi)
tit_w2v = dict(TITLE.vocab.stoi)
art_v2w = {}
tit_v2w = {}
for (a_k,a_v),(t_k,t_v) in zip(art_w2v.items(),tit_w2v.items()):
    art_v2w[a_v] = a_k
    tit_v2w[t_v] = t_k


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
train_data,test_data = data.BucketIterator.splits((train,test),
                    batch_sizes=(32,32),device = device,repeat=False,sort_within_batch = True,
                sort_key = lambda x : len(x.article),)

INPUT_DIM = len(ARTICLE.vocab)
OUTPUT_DIM = len(TITLE.vocab)

ENC_EMB_DIM = 512
DEC_EMB_DIM = 512
ENC_HID_DIM = 256
DEC_HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = ARTICLE.vocab.stoi['<pad>']
SOS_IDX = TITLE.vocab.stoi['<sos>']
EOS_IDX = TITLE.vocab.stoi['<eos>']

attn = models.Attention(ENC_HID_DIM, DEC_HID_DIM)
encoder = models.Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
decoder = models.Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = models.Seq2Seq(encoder, decoder, device,PAD_IDX,SOS_IDX,EOS_IDX).to(device)
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
model.apply(init_weights)
opt = optim.Adam(model.parameters())
#opt = adabound.AdaBound(model.parameters(),lr = 0.1)

pad_idx = TITLE.vocab.stoi["<pad>"]
loss_f = nn.CrossEntropyLoss(ignore_index=pad_idx)

def train(model,optimizer, loss_f,clip):
    model.train()
    run_loss = 0
    for i,batch in enumerate(train_data):
        sentence,seq_len = batch.article
        title = batch.title
        optimizer.zero_grad()
        output,attention = model(sentence, title,seq_len )
        output = output[1:].view(-1, output.shape[-1])
        title = title[1:].view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]
        
        loss = loss_f(output, title)
        
        loss.backward()
        
        th.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        run_loss += loss.detach().cpu()
        
    return run_loss / len(train_data)

def test(model,loss_f):
    model.eval()
    run_loss = 0
    with th.no_grad():
        for batch in test_data:
            sentence,seq_len = batch.article
            title = batch.title

            output,_ = model(sentence, title,seq_len)
            output = output[1:].view(-1, output.shape[-1])
            title = title[1:].view(-1)
            loss = loss_f(output, title)
            run_loss += loss.detach().cpu()

    return run_loss / len(test_data)

import time
s = time.time()
train_losses,test_losses=[],[]
for epoch in range(90):
    train_loss = train(model, opt, loss_f, 1)
    test_loss = test(model,loss_f)
    print(epoch,train_loss,test_loss)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
print(time.time()-s)
th.save(model.state_dict(), './save_model/title_generate.pth')
