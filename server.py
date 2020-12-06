import os

from flask import Flask, request
from flask import render_template

import torch
from model import Model
from dataset import Dataset


def predict(dataset, model, text, total_words=24):
    words = list(text)
    words_len = len(words)
    model.eval()

    # 手动设置第一个词为<START>
    input = torch.Tensor([dataset.word_to_index['<START>']]).view(1, 1).long()
    input = input.cuda()
    hidden = None

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
            # 转为字
            w = dataset.index_to_word[top_index]

            # 追加到words中
            words.append(w)
            # 拼接到input中继续传递
            input = input.data.new([top_index]).view(1, 1)

        if w == '<EOP>':
            del words[-1]
            break

    return '\n' + '\n'.join(''.join(words).split('。')) + '\n'


# model加载
dataset = Dataset({'sequence_length': '48'})
model = Model(dataset)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 数据加载
model.load_state_dict(torch.load('model_data/model_99.pth'))

# ========================================================================
# web server 准备
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def result():
    # 处理上传文件
    file = request.files['image']
    if len(file.filename) == 0:
        return 'No image, select a image first...'
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # 传参获取
    prefix = request.form.get('prefix') if request.form.get('prefix') else '床前明月光，'
    # 上传文件存储路径
    path = app.config['UPLOAD_FOLDER'] + '/' + filename
    # model输出结果
    output = predict(dataset, model, text=prefix, total_words=120)

    return render_template('result.html', data={'result': output, 'path': path})


# 开启server
app.run(host='0.0.0.0', port=8080)
