import argparse
import sys

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset


def train(dataset, model, args):
    # 设置为训练模式
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调整
    criterion = nn.CrossEntropyLoss()

    dataloader = DataLoader(
        dataset,
        # shuffle=True,
        batch_size=args.batch_size,
    )
    for epoch in range(args.max_epochs):
        if (args.previous != 0) and (epoch <= args.previous):
            print('continue...', epoch)
            continue
        # state_h, state_c = model.init_state(args.sequence_length)

        for batch, (input, label) in enumerate(dataloader):
            input, label = input.to(device), label.to(device)
            # 因为outputs经过平整，所以labels也要平整来对齐
            label = label.view(-1)

            # 初始为0，清除上个batch的梯度信息
            optimizer.zero_grad()

            output, (state_h, state_c) = model(input)
            loss = criterion(output, label)

            # state_h = state_h.detach()
            # state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            print({'epoch': epoch, 'batch': batch, 'loss': loss.item()})
            # print('now result: ', predict(dataset, model, text='你说什么？', total_words=15))
            # model.train()
            # sys.exit()

        # model.eval()
        print('epoch result-----------:', predict(dataset, model, text='窗前明月光，', total_words=48))
        model.train()
        print('saving epoch...', epoch)
        torch.save(model.state_dict(), 'model_data/%s_%s.pth' % ('model', epoch))
    print('Train Finished...........')


def predict(dataset, model, text, total_words=24):
    # words = text.split(' ')
    words = list(text)
    words_len = len(words)
    model.eval()

    # 手动设置第一个词为<START>
    input = torch.Tensor([dataset.word_to_index['<START>']]).view(1, 1).long()
    # input = torch.Tensor([dataset.word_to_index['天']]).view(1, 1).long()
    input = input.cuda()
    hidden = None

    # 初始化input hidden
    # if words:
    #     for word in words:
    #         output, hidden = model(input, hidden)
    #         print(word)
    #         input = input.data.new([dataset.word_to_index[word]]).view(1, 1)

    # state_h, state_c = model.init_state(len(words))

    # 补齐剩下的字
    # for i in range(0, total_words - words_len):
    for i in range(total_words):
        output, hidden = model(input, hidden)
        # 提供的前几个字 直接拼接
        if i < words_len:
            w = words[i]
            input = input.data.new([dataset.word_to_index[w]]).view(1, 1)
        else:
            # 预测字的索引
            top_index = output.data[0].topk(1)[1][0].item()
            # print('index is:', top_index)
            # 转为字
            w = dataset.index_to_word[top_index]
            # print('word is:', w)

            # 追加到words中
            words.append(w)
            # 拼接到input中继续传递
            input = input.data.new([top_index]).view(1, 1)

        if w == '<EOP>':
            del words[-1]
            break

    return '\n' + '\n'.join(''.join(words).split('。')) + '\n'

# 藏头诗
def predict_head(dataset, model, text, total_words=24):
    # words = text.split(' ')
    words = list(text)
    words_len = len(words)
    model.eval()

    pre_word = '<START>'
    result = []
    head_index = 0
    # 手动设置第一个词为<START>
    input = torch.Tensor([dataset.word_to_index[pre_word]]).view(1, 1).long()
    # input = torch.Tensor([dataset.word_to_index['天']]).view(1, 1).long()
    input = input.cuda()
    hidden = None

    # 补齐剩下的字
    # for i in range(0, total_words - words_len):
    for i in range(total_words):
        output, hidden = model(input, hidden)

        if (pre_word in ['。', '！', '<START>']):
            w = words[head_index]
            result.append(w)
            head_index += 1
            input = input.data.new([dataset.word_to_index[w]]).view(1, 1)

        else:
            # 预测字的索引
            top_index = output.data[0].topk(1)[1][0].item()
            # 转为字
            w = dataset.index_to_word[top_index]
            # 追加到words中
            result.append(w)
            # 拼接到input中继续传递
            input = input.data.new([top_index]).view(1, 1)

        # 记录上一个字
        pre_word = w
        # 藏头完全包括了输入
        if head_index == words_len:
            break

        if w == '<EOP>':
            del result[-1]
            break

    # return ' '.join(result)
    return "\n" + "\n".join(''.join(result).split('。'))


parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
# parser.add_argument('--sequence-length', type=int, default=48)
parser.add_argument('--sequence-length', type=int, default=256)
parser.add_argument('-t', '--test', type=str, default='')
parser.add_argument('-txt', type=str, default='天下无人')
parser.add_argument('-previous', type=int, default=0)

args = parser.parse_args()

# print(args.txt)
# sys.exit()

dataset = Dataset(args)
# print(dataset, dataset[1])
# sys.exit()
model = Model(dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 预测模式
if args.test == 'test':
    # 加载数据
    # model.load_state_dict(torch.load('model_data/model_0.pth'))
    model.load_state_dict(torch.load('model_data/trained100.pth'))
    print(predict(dataset, model, text=args.txt, total_words=120))

else:
    # model.train()
    # 继续之前的训练
    if args.previous != 0:
        pth_file = 'model_data/%s_%s.pth' % ('model', args.previous)
        print('loading previous pth file: ', pth_file)
        model.load_state_dict(torch.load(pth_file))
        # print('previous result:', predict(dataset, model, text='你说什么？', total_words=48))

    train(dataset, model, args)
    # print(predict(dataset, model, text='床前', total_words=12))
