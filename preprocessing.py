import os
from pathlib import Path
import re
import neologdn
import numpy as np
import janome
# タイトル、記事、メディアを収集
titles, articles, labels = [], [], []
#news_list = ['dokujo-tsushin' ,'it-life-hack', 'kaden-channel', 'livedoor-homme', 'movie-enter', 'peachy', 'smax', 'sports-watch', 'topic-news']

news_list = ['topic-news']
for i, media in enumerate(news_list):
    files = os.listdir(Path("text",media)) #ディレクトリ下の全てのファイル抽出
    for file_name in files:
        if file_name == 'LICENSE.txt':
            continue
        with Path('text', media, file_name).open(encoding='utf-8') as f:
            lines = [line for line in f]
            title = lines[2].replace('\n', '')
            text = ''.join(lines[3:])
            titles.append(title)
            articles.append(text.replace('\n', ''))
            labels.append(i)


def format_text1(text):
    text = neologdn.normalize(text)
    text=re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-…]+', "", text)
    text= re.sub(r'\d+', '0', text)
    text = re.sub(r'[!-/:-@[-`{-~]', r'', text)
    text = re.sub(u'[■-♯]', '', text)
    text = text.replace("・","")
    text = re.sub("【関連(情報|記事)】.*","",text)
    text = text.replace('「',"").replace('」',"").replace('「',"").replace('【',"").replace('】',"").replace('・',"").replace('…',"")
    return text
    
article_f = [format_text1(t) for t in article_]
title_f = [format_text1(t) for t in title_]

# 文の長さの調整
len_sentence = [len(s) for s in article_f]
len_sentence = np.array(len_sentence)
article_ff,title_ff = [],[]
for i,t in enumerate(len_sentence):
    if 350<t<750:
        article_ff.append(article_f[i])
        title_ff.append(title_f[i])

# "article"\t"title" の行の形で tsv形式で保存
tmp = list(map(lambda x:"\t"+x,title_ff))
data = [i+j for i,j in zip(article_ff,tmp)]
idx = len(data) - 100
train,test = data[:idx],data[idx:]
with open("./data/train.tsv", mode='x') as f:
    f.write('\n'.join(train))
with open("./data/test.tsv", mode='x') as f:
    f.write('\n'.join(test))
