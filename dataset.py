import re
import sys

import torch
import pandas as pd
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        # 所有字
        self.words = self.load_words()
        # 去重
        self.uniq_words = self.get_uniq_words()
        # print(len(self.words), len(self.uniq_words), self.uniq_words[0:100])
        # sys.exit()

        # 追加起止符和空格
        self.uniq_words.extend(['<EOP>', '<START>', '</s>'])
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        # 所有诗句字转为索引表示
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv('data/tang-poem5.txt', squeeze=True)
        # 拼接成长句
        txt = train_df.str.cat(sep='')
        return txt

        # f = open("data/the_arabian_nights.txt", "r")
        # txt = f.read()
        # return txt

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        # 获取总条数\页数
        # return len(self.words_indexes) // self.args.sequence_length
        # 按字逐个移动
        return len(self.words_indexes) - self.args.sequence_length

    def __getitem__(self, index):
        # Shuffle 随机是通过index随机跳动来实现的
        sequence_length = self.args.sequence_length

        # input和tag分别为
        # <START> 1 2 3
        #    1    2 3 <EOP>
        # input = [self.word_to_index['<START>']] + self.words_indexes[index * sequence_length:(index + 1) *sequence_length]
        # tag = self.words_indexes[index * sequence_length:(index + 1) *sequence_length] + [self.word_to_index['<EOP>']]

        # print([self.index_to_word[w] for w in input],"\n" ,[self.index_to_word[w] for w in tag])
        # sys.exit()

        # return (torch.tensor(input), torch.tensor(tag))

        # return (
        #     # 输入
        #     torch.tensor(self.words_indexes[index * sequence_length:(index + 1) *sequence_length]),
        #     # 后移一个字作为tag
        #     torch.tensor(self.words_indexes[index * sequence_length + 1:(index + 1) * sequence_length + 1]),
        # )

        # 逐字移动
        return (
            torch.tensor(self.words_indexes[index:index + self.args.sequence_length]),
            torch.tensor(self.words_indexes[index + 1:index + self.args.sequence_length + 1]),
        )
