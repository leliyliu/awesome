
- [awesome](#awesome)
    - [DL ON FPGA](#DL_ON_FPGA)
    - [论文阅读](#paper)
    - [英语学习](#english)
    - [algorithms & paper](#algorithms)
    - [book & course](#book_course)
    - [龙芯杯](#龙芯杯)
    - [Go语言](#Go语言)
    - [操作系统](#操作系统)
    - [tools](#tools)
    - [Interesting](#Interesting)


# <span id = "awesome">awesome</span>
manage my star projects on github  


刚开学，想着要规划一下后面的任务，于是开始了一波整理，今天突然意识到已经在github 上star 了不少项目，有时候想用的时候却找不到了，上网查了一下，发现相关的管理插件和网页也都不太好用，看到一个说法，如果都不愿意整理的star 项目，star起来确实也没什么用，因此还是决定好好整理一下，顺便筛查一波。

## DL_ON_FPGA
+ 由于最近做的工作主要是关于在FPGA上做深度学习加速，因此在这里记录一下相关的工作，有一些做的比较好的可以借鉴。

### [hls4ml](https://github.com/hls-fpga-machine-learning/hls4ml)
Machine learning in FPGAs using HLS。一个实现了machine learning 常规算法的hls包， translate traditional open-source machine learning package models into HLS 。

### [TF2](https://github.com/TF2-Engine/TF2)
An Open Source Deep Learning Inference Engine Based on FPGA。这应该是浪潮科技最近刚开源的一个基于FPGA的深度学习加速开源工具，有很大的价值。可以移植caffe,tensorflow,pytorch等等。但是这是基于intel 的opencl的，和HLS还有一定的区别，主要借鉴一下即可。

### [Jaffe](https://github.com/jiangwx/Jaffe)
某个人做的caffe的底层C代码的开源。

### [pp4fpgas-cn](https://github.com/xupsh/pp4fpgas-cn)
中文版 Parallel Programming for FPGAs.HLS工程源代码: https://github.com/xupsh/pp4fpgas-cn-hls

### [FPGA-ZynqNet](https://github.com/pp-Innovate/FPGA-ZynqNet)
FPGA-based ZynqNet CNN accelerator developed by Vivado_HLS\
基于FPGA 构建的ZynqNet CNN 加速器，通过Vivado_HLS 来综合实现

### [zynqnet](https://github.com/dgschwend/zynqnet)
基于Zynq 实现的相应的CNN加速器的具体内容。

## <span id = "paper">论文阅读</span>
+ 主要包括了一些深度学习相关的论文阅读计划之类的，主要是一些经典论文的推荐和阅读列表。

### [state-of-the-art-result-for-machine-learning-problems](https://github.com/RedditSota/state-of-the-art-result-for-machine-learning-problems)
一些经典的深度学习项目和论文，包括NLP:语言模型，机器翻译，文本分类，自然语言交互，智能问答，命名实体识别，Abstractive Summarization，Dependency Parsing; CV：Classification，Instance Segmentation，Visual Question Answering，Person Re-identification; Speech:ASR;半监督，无监督学习，迁移学习，强化学习。

### [Deep-Learning-Papers-Reading-List](https://github.com/ycszen/Deep-Learning-Papers-Reading-List)
主要是CV中的一些相关论文，包括了识别，分割和分类的一些经典论文，同时还包括了一些数据集和软件推荐。

### [Deep-Learning-Papers-Reading-Roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
论文阅读计划，但是只更新到17年。可以略读了解。

### [GNNPapers](https://github.com/thunlp/GNNPapers)
GNN相关经典paper

## <span id = "english">英语学习</span>
+ 主要是几个程序员分享的英语学习的经验性的github项目，值得借鉴。（具体内容不加介绍，看即可）

### [English-level-up-tips-for-Chinese](https://github.com/byoungd/English-level-up-tips-for-Chinese)

### [A-Programmers-Guide-to-English](https://github.com/yujiangshui/A-Programmers-Guide-to-English)


## <span id = "algorithms">algorithms & paper</span>
+ 有关算法的一些项目，包括了一些传统的算法课程和机器学习相关算法介绍的一个综合合集。同时，还包括了一些论文的介绍和复现源码。
### [algorithms](https://github.com/jeffgerickson/algorithms)
普林斯顿大学经典的算法课程的课本教材，是由Jeff 出版的一本算法书籍，和MIT的算法导论一样经典，而且普林斯顿大学的教程更注重编程。

### [fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)
伯克利大学一个Fully Convolutional Networks for Semantic Segmentation的介绍，和具体代码。

### [Python-100-Days](https://github.com/jackfrued/Python-100-Days)
100天快速掌握python的大多数应用方面。

### [neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)
一篇利用深度学习进行推荐系统研究的论文。

### [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
Mask_RCNN源码

### [models](https://github.com/tensorflow/models)
tensorflow官方给出的内置的tensorflow的相关模型，用于熟悉深度学习。

### [BTS-DSN](https://github.com/guomugong/BTS-DSN)
眼底血管分类分割研究

### [LeetCode](https://github.com/strengthen/LeetCode)
leetcode解题合集

### [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics)
一些算法合集（主要是机器人控制的算法）

### [YOLO-Reproduce-Summary](https://github.com/amusi/YOLO-Reproduce-Summary)
YOLO源码合集

### [deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)
物体检测论文合集

### [DataMiningAlgorithm](https://github.com/linyiqun/DataMiningAlgorithm)
数据挖掘18大算法实现以及其他相关经典DM算法


## book_course
+ 主要收集了一些相关的教程书籍，包括机器学习，深度学习等一系列方面的教材和相关教程。
### [reinforcement-learning](https://github.com/dennybritz/reinforcement-learning)
强化学习的算法实现，与 Sutton's Book and David Silver's course配套。关于这一课程更具体的信息可以参考[强化学习](http://www.wildml.com/2016/10/learning-reinforcement-learning/)，在项目介绍中，你可以找到相关书籍和课程。

### [PyTorchDocs](https://github.com/fendouai/PyTorchDocs)
pytorch 官方中文教程，包括了计算机视觉，自然语言处理，生成对抗网络，强化学习......
可以多多了解，毕竟现在做学术的话，用的最多的还是Pytorch。

### [gold-miner](https://github.com/xitu/gold-miner)
一个翻译平台，主要翻译一些相关技术文档。

### [pandas-cookbook](https://github.com/jvns/pandas-cookbook)
pandas的一个学习文档

### [django2.0-course](https://github.com/HaddyYang/django2.0-course)
django 学习

### [awesome-tensorflow](https://github.com/jtoy/awesome-tensorflow)
一个tensorflow资料合集

### [Virgilio](https://github.com/virgili0/Virgilio)
一个在线的机器学习指导教程。

### [pydata-notebook](https://github.com/BrambleXu/pydata-notebook)
利用Python进行数据分析 第二版 (2017) 中文翻译笔记

### [pumpkin-book](https://github.com/datawhalechina/pumpkin-book)
《机器学习》（西瓜书）公式推导解析，在线阅读地址：https://datawhalechina.github.io/pumpkin-book

### [Tensorflow-](https://github.com/MachineLP/Tensorflow-)
Tensorflow实战学习笔记、代码、机器学习进阶系列

### [cs231n.github.io](https://github.com/cs231n/cs231n.github.io)
斯坦福cs231n课程

### [Awesome-System-for-Machine-Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning) 
系统的机器学习教程（资源合集）

### [mml-book.github.io](https://github.com/mml-book/mml-book.github.io)
Companion webpage to the book "Mathematics For Machine Learning"(机器学习数学基础)

### [DeepLearning-500-questions](https://github.com/scutan90/DeepLearning-500-questions)
深度学习500问，以问答形式对常用的概率知识、线性代数、机器学习、深度学习、计算机视觉等热点问题进行阐述，以帮助自己及有需要的读者。

### [AiLearning](https://github.com/apachecn/AiLearning)
AiLearning: 机器学习 - MachineLearning - ML、深度学习 - DeepLearning - DL、自然语言处理 NLP 
资源合集

## 龙芯杯
+ 这里主要包括了一些参加龙芯杯的优秀队伍的项目，包括了第一二三届，前两届由于之前总结过，所以我会放在另一个markdown文档中。
[龙芯杯参考资料](https://github.com/leliyliu/awesome/blob/master/%E9%BE%99%E8%8A%AF%E6%9D%AF%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99.md)
### [nscscc2019ucas](https://github.com/nscscc2019ucas/nscscc2019ucas)
国科大2019年龙芯杯参赛作品

### [nontrivial-mips](https://github.com/trivialmips/nontrivial-mips)
清华2019龙芯杯参赛作品

### [MangoMIPS32](https://github.com/RickyTino/MangoMIPS32)
哈工大威海2019参赛作品

### [verilog-ethernet](https://github.com/alexforencich/verilog-ethernet)
利用FPGA实现以太网的接口，包括ipv4的接通等等。

### 其它龙芯杯参赛作品合集
+ [one](https://github.com/cquca/nscscc_2018)
+ [two](https://github.com/MaxXSoft/Uranus)
+ [three](https://github.com/kjayawant/BranchPrediction)
+ [four](https://github.com/patc15/mipscpu)
+ [five](https://github.com/valar1234/MIPS)

### [MIPS-OS](https://github.com/BXYMartin/MIPS-OS)
好像是用MIPS 移植JOS的工作，可以学习一下

## Go语言
+ 由于最近Go语言比较火，其主要是运用在区块链之类的新兴技术上，所以比较感兴趣，了解了一下go语言编程，把相关的项目也star了一下，有机会可以尝试多写一写go 的相关程序。
### [go-fundamental-programming](https://github.com/unknwon/go-fundamental-programming)
《Go 编程基础》是一套针对 Google 出品的 Go 语言的视频语音教程，主要面向新手级别的学习者。
### [build-web-application-with-golang](https://github.com/astaxie/build-web-application-with-golang)
如何利用go语言搭建一个web开发端，这是一个实践性的项目，而且非常有心，出了很多不同的语言版本，还可以通过此来锻炼自己的外语。
### [the-way-to-go_ZH_CN](https://github.com/unknwon/the-way-to-go_ZH_CN)
《The Way to Go》中文译本，中文正式名《Go 入门指南》，看就完事。

## 操作系统
+ 由于以后其实主要还是想从事系统方面的工作，在这方面准备也是比较多的，这边暂时收集了清华的一些操作系统课的资料

### [ucore-os-learning](https://github.com/jackwener/ucore-os-learning)
清华本科操作系统课基本介绍和相关资源分享。

### [simple_os_book](https://github.com/chyyuu/simple_os_book)
一本有趣的操作系统实践书籍，主要还是ucore

### [my_ucore_os_lab](https://github.com/markjenny/my_ucore_os_lab)
一个自己做了ucore实验的人的开源项目，可以做了之后参考对比。

### [ucore实验指导书](https://github.com/chyyuu/ucore_os_docs)
清华ucore操作系统实验指导

### [ucore_os_lab](https://github.com/chyyuu/ucore_os_lab)
ucore实验

### [AOS_Paper_reading](https://github.com/m0xiaoxi/AOS_Paper_reading)
清华研究生操作系统课程和相关论文研读。

### [AIMv6](https://github.com/davidgao/AIMv6)
Cross-platform operating system for teaching purposes. 一个交叉的操作系统教学平台

### [mit6.828-2017](https://github.com/shishujuan/mit6.828-2017)
jos操作系统2017年相关答案，同时包括了stju 的相关exercise

## tools
+ 这里主要包含了一些工具，主要是一些相关的插件和python库的介绍，同时也包含一些其它的工具。
### [tflearn](https://github.com/tflearn/tflearn)
tflearn 是tensorflow 所提供的一个高级的API接口。

### [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
由于当前使用的最多的还是pytorch和tensorflow这两个深度学习开发平台，因此将caffe代码转为tensorflow代码是个不错的选择。

### [Surprise](https://github.com/NicolasHug/Surprise)
python 所提供的一个机器学习的推荐系统相关的库，里面包括了SVD等基本的算法。

### [freecodecamp.cn](https://github.com/FreeCodeCampChina/freecodecamp.cn)
一个前端开发的练习网站

### [3DGNN_pytorch](https://github.com/yanx27/3DGNN_pytorch)
This is the Pytorch implementation of 3D Graph Neural Networks for RGBD Semantic Segmentation
三维重建

### [Algorithm_Interview_Notes-Chinese](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese)
2018/2019/校招/春招/秋招/算法/机器学习(Machine Learning)/深度学习(Deep Learning)/自然语言处理(NLP)/C/C++/Python/面试笔记
用于练习

## Interesting
+ star了一些有趣的项目，这个分类里面比较杂，但是不多，因此就不多作介绍了。
### [themostdangerouswritingapp](https://github.com/maebert/themostdangerouswritingapp)
一个蛇皮编辑器，如果在规定时间内没有进行操作，那么将前功尽弃。
### [CUMCMThesis](https://github.com/latexstudio/CUMCMThesis)
数学建模国赛论文latex 模板。
### [ds-cheatsheets](https://github.com/FavioVazquez/ds-cheatsheets)
数据科学编程语言的速查表
### [GitHubDaily](https://github.com/GitHubDaily/GitHubDaily)
推荐github项目的一个平台
### [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet)
Latex code for making neural networks diagrams
### [MikuTools](https://github.com/Ice-Hazymoon/MikuTools)
一个轻量的工具集合 http://tools.imiku.me
### [ShadowSocksShare](https://github.com/the0demiurge/ShadowSocksShare)
爬虫，翻墙share
### [GeneralConfig](https://github.com/yuhaowow/GeneralConfig)
all config file for code(配置文件)
### [TensorFLow-Learning](https://github.com/Parker-Lyu/TensorFLow-Learning)
B站上炼数成金的公开课笔记
### [2019-false-news-detection-challenge](https://github.com/deping-1/2019-false-news-detection-challenge)
智源研究院&中科院计算所-互联网虚假新闻检测挑战赛，一个开源代码。
### [awesome-resume-for-chinese](https://github.com/dyweb/awesome-resume-for-chinese)
中文简历模板,latex