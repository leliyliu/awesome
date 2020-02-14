---
title: 动手学深度学习 pytorch 1
date: 2020-2-14 13:16:38
tags: 动手学深度学习 pytorch
categories: 动手学深度学习
keywords: 
description:
top_img: 
comments  
cover:  
toc:  true
toc_number: 
copyright: true
---



[TOC]

# 动手学深度学习 task1

---

## 线性回归

对于简单的线性回归而言，实际上就非常简单，只需要一个线性层即可。对于python而言，一般来说，使用科学计算提供的矢量运算要比利用循环实现的效率高很多。这里实际上跟CPU本身的指令集扩展有关，包括AVX和MME的矢量运算指令。

### yield 使用

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # random read 10 samples
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # the last time may be not enough for a whole batch
        yield  features.index_select(0, j), labels.index_select(0, j)
```

在这里，使用了yield，这里的yield 有两个作用，包括return 和一个迭代器的作用。在python 中，这样的例子还包括之前有的xrange，即现在的range，如果去打印type(range(10))，可以看到，当前的range返回的也是这样一个迭代器。

### 小批量计算

batch normalization 是两种不同方式的中和，既不需要计算整个批量的所有数据来进行梯度下降，也不由于某些极少数的点来导致下降的方向不对。一般而言，小批量的计算是用于对于大量数据的梯度下降， 而对于少量的数据，直接使用批量梯度下降即可。

### 反向传播

一般而言，神经网络都是BP神经网络，也就是反向传播是很重要的一个过程，故而在训练的过程中，需要保持参数的反向传播（能计算其导数），在pytorch中，给出了相应的方法来进行反向传播。

## softmax 与分类模型

### softmax

softmax的提出，很大程度上是为了数值计算的方便。对于softmax的理解，主要要通过信息论和数值计算这两方面来进行，softmax函数本身所具有的特性，让他能够很好地运用在分类网络中。

### 交叉熵

对于交叉熵，这里实际上是信息论的一个重要概念，其本身用在分类模型中，表示我们只关心分类出来的那个结果的概率比的大小，而不在意其它的结果。

## 多层感知机

多层感知机其原理就是通过在一层和一层之间加上激活函数，使得本来很简单的过程可以映射到更复杂的网络结构和高维的数据分布结构当中。

## 循环神经网络

### 两种不同采样方式下的处理

在实现RNN 的训练过程中，有这样一段代码：

```python
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  # 否则需要使用detach函数从计算图分离隐藏状态
                for s in state:
                    s.detach_()
           

```

对于两种不同的采样方式，采用了不同的对于隐藏状态H 的处理方式，对此进行理解

#### 随机采样

对于随机采样，由于采样结果的不连续，那么之前所具有的状态H对当前是没有意义的，故需要重新初始化状态``init_rnn_state`。

#### 相邻采样

在相邻采样开始前，首先进行了`init_rnn_state` 函数调用，即进行了相应的初始化。由于使用的是相邻采样，那么正如例子所展示的那样，得到的数据应该是相邻连续的：

```
X:  tensor([[ 0,  1,  2,  3,  4,  5],
        [15, 16, 17, 18, 19, 20]]) 
Y: tensor([[ 1,  2,  3,  4,  5,  6],
        [16, 17, 18, 19, 20, 21]]) 

X:  tensor([[ 6,  7,  8,  9, 10, 11],
        [21, 22, 23, 24, 25, 26]]) 
Y: tensor([[ 7,  8,  9, 10, 11, 12],
        [22, 23, 24, 25, 26, 27]]) 
```

所以实际上，进行`data_iter_fn`迭代的时候，所得到的数据，即是连续的，那么其本身的状态是可以保持的，正如用[0,1,2,3,4,5]的状态推出到[1,2,3,4,5,6]的状态一样，这样的过程是连续进行的，所以状态也应该是连续的。所以这个时候，应该保持状态H不变，而不需要像随机采样那样重新`init_rnn_state`，所以我们需要使用detach_来保留当前H的状态，来进行后续的推导，那么为什么要使用`detach_`而不直接保留呢？

可以参见detach_的源码：

```python
# detach_ 的源码
def detach_(self):
    """Detaches the Variable from the graph that created it, making it a
    leaf.
    """
    self._grad_fn = None
    self.requires_grad = False
```

可以看到，这里将其将 Variable 的grad_fn 设置为 None，这样，BP 的时候，到这个 Variable 就找不到 它的 grad_fn，所以就不会再往后BP了。

因为采样的时候，已经预设了相应的前提，即确定的时间步长，不应该再向前BP（反向传播），故而需要将其detach掉。

对于detach_函数的理解，可以参考这篇博客[pytorch中的detach\_和detach](https://www.cnblogs.com/jiangkejie/p/9981707.html)

