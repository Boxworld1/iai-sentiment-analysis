# 模型使用方法



## 可选参数

|    命令     | 参数          | 用途                                     |
| :---------: | :------------ | :--------------------------------------- |
|  -m <str>   | MODEL         | 选择模型，可选参数: cnn, mlp, lstm, gru  |
|  -bs <int>  | BATCH_SIZE    | 每批次训练的数据规模                     |
|  -hs <int>  | HIDDEN_SIZE   | 隐藏层/ 卷积核个数                       |
|  -sl <int>  | SENTENCE_LEN  | 句子词数                                 |
|  -ep <int>  | EPOCH         | 最大迭代次数                             |
| -lr <float> | LEARNING_RATE | 学习率                                   |
|  -t <str>   | TRAIN         | 是否训练新模型，可选参数: Y/N            |
|  -g <int>   | GPU_ID        | 若有 GPU 资源，则可选择要使用的 GPU 编号 |



## 使用教学

**请确保你是在文件根目录下开启命令行，然后在命令行输入：**

```
python main.py
```

这时默认使用的模型是隐藏层大小为 64 的 MLP 模型。 

你也可以尝试下面的代码来使用已经训练好的模型：

```
python main.py -m cnn -t N
python3 main.py -m lstm -hs 128 -t N
python3 main.py -m gru -lr 0.01 -t N
```

需要注意的是，即使是相同的已经训练好的模型，由于句子词数不足时会利用 50 维的随机向量作填充，因此每一次测试的结果有所不同是十分正常的现象，如果出现准确率极低的情况，那么重跑一次代码即可。