# LSTM 唐诗生成

> 深度神经网络学习打卡，基于五言绝句训练集，训练一个可以创作唐诗的机器模型

## 执行环境

- ubuntu16.04
- python3
- pytorch 1.7
- cuda 10.1


## 训练

> Geforce 1080 8G, 20epoch，耗时6h，如果GPU内存强悍的话可以调大`--batch-size`

```bash
python train.py --max-epochs=20 --batch-size=256
```


## 生成

> 给定`-txt`提示词，即可自动生成唐诗

```bash
python train.py -t test -txt 秋色见浮生，
```

效果如下

```
秋色见浮生，空思白浪中
此时寻古墓，揖佛不能言
月黑霜霜动，春来曳滴浓
只应随范蠡，相忆向南朝
澹然遭报岁，销即脱金郎
只是同人劝，凄凉札簟寒
圣人应律利，吾子格斯梁
雪作擎山果，松多认雪松
验书成洞口，何事是尘埃
塔作青山险，年来白浪魂
```
