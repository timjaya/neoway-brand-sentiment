# Author: Felipe Campos Penha

import pandas as pd
import re

# Ref: https://github.com/delver/wordnet-db/blob/master/dict/dbfiles/noun.food

filename = 'noun.food'

with open(filename, 'r') as file:
    data = file.read().replace('\n', '')

regex = '((\#[a-zA-Z0-9_-]*)|([a-zA-Z0-9_-]*\,\@)|([a-zA-Z0-9.]*[:]\w*\,\@)|(\[([a-zA-Z0-9+.:]*).+?(?=\])\])|(\(([^()]*|\([^()]*\))*\))|(\{\s)|(\s\})|([a-zA-Z0-9.]*[:]\w*[,;]*[a-zA-Z0-9]*)|[0-9])'

data_p1 = data.upper()

data_p1 = re.sub(regex, '', data_p1)

data_p1 = re.sub('(\_)', ' ', data_p1)

data_p1 = re.sub('(\-)', ' ', data_p1)

data_p1 = re.sub('(\s+)', ' ', data_p1)

data_p1 = re.sub('\}', ' ', data_p1)

regex = '((\#[a-zA-Z0-9_-]*)|(\@)|([0-9])|([a-zA-Z0-9.]*[:]\w*\,\@)|(\[([a-zA-Z0-9+.:]*).+?(?=\])\])|(\(([^()]*|\([^()]*\))*\))|(\{\s)|(\s\})|([a-zA-Z0-9.]*[:]\w*[,;]*[a-zA-Z0-9]*)|[0-9])'

data_p2 = data.upper()

data_p2 = re.sub(regex, '', data_p2)

data_p2 = re.sub('(\_)', ' ', data_p2)

data_p2 = re.sub('(\-)', ' ', data_p2)

data_p2 = re.sub('(\s+)', ' ', data_p2)

data_p2 = re.sub('\}', ' ', data_p2)

lst_p1 = list(set(data_p1.split(', ')))

lst_p2 = list(set(data_p2.split(', ')))

lst = list(set(lst_p1 + lst_p2))

while('' in lst): 
    lst.remove('')

df = pd.DataFrame(lst, columns=['word'])

filename = 'wordnet_food_beverages_list'

df.to_csv(filename,
              sep=',',
              index=False
             )

