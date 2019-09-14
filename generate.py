#!pip install japanize_matplotlib 
import japanize_matplotlib 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchtext import data
from janome.tokenizer import Tokenizer
import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
import torch.nn.functional as F
from train import read_data
import models
def generate(model, sentence):
    model.eval()
    tokenize = _tokenizer(format_text1(sentence))
    sentence = ["<sos>"] + [w for w in tokenize] + ["<eos>"]
    w2v_in = [ARTICLE.vocab.stoi[w] for w in sentence]
    length = th.LongTensor([len(w2v_in)]).to(device)
    w2v_in = th.LongTensor(w2v_in).unsqueeze(1).to(device)
    output, attention = model(w2v_in, None, length)
    #output = F.softmax(output,dim = 2)
    pred = th.argmax(output.squeeze(1),1)
    pred = [TITLE.vocab.itos[w] for w in pred]
    pred, attention = pred[1:],attention[1:]
    return pred, attention
def display_attention(article, title, attention):
    
    fig = plt.figure(figsize=(100,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
    ax.tick_params(labelsize=15)
    ax.set_xticklabels([''] + ['<sos>'] + [t for t in _tokenizer(article)] + ['<eos>'], 
                       rotation=45)
    ax.set_yticklabels([''] +title)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


ARTICLE, TITLE, train,test = read_data()

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
attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
encoder = models.Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
decoder = models.Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = models.Seq2Seq(encoder, decoder, device,PAD_IDX,SOS_IDX,EOS_IDX).to(device)
weigth = th.load("save_model/title_generate.pth")
model.load_state_dict(weigth)
article = "7月2日,プライベート写真が流出したことでAKB48としての活動を辞退した米沢瑠美が、新しい事務所が決まったことを自身のツイッターで明かした。米沢は7月1日、「みんなに早く伝えたかった事を、話せる準備が整ってきましたっ☆ まず、所属事務所のご報告。エムズエンタープライズさんに所属することになりました☆」と報告。今年3月いっぱいで所属事務所との契約が満了したため、約2年間続いたブログを閉鎖することとなった米沢だが、今回事務所が決まったことで、新たなオフィシャルブログを製作中。今月中旬頃にはスタートする予定だという。また、「これからは演技のお仕事を中心に頑張っていきたいと思っております(^^)」と今後の方針を示唆。どんどん活動の場を広げると思われる米沢から、今後も目が離せそうにない。"
idx_ = 90 # article に対応するtitleのindex
#article = "".join(vars(train.examples[idx_])["article"])
true = "".join(vars(train.examples[idx_])["title"])
pred_title, attention = generate(model,article)
print("".join(article))
print("[predict]","".join(pred_title))
print("[true]",true)
display_attention(article,pred_title,attention)
