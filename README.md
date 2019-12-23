# <font size=4>NLP Pytorch Template</font>

&emsp;适用于常见的NLP任务，比如文本分类，文本匹配，多轮QA检索等，可以快速搭建自己的模型并进行训练。基于[MatchZoo]( https://github.com/NTMC-Community/MatchZoo )进行修改，重构文件目录，自定义了中文的预处理器以及适用于多轮QA的padding函数，所以模型搭建的基本流程和MatchZoo一致。



- `snlp`：主模块文件，用于调用。
- `sample_data`：示例数据。
- `tests`：tutorial，使用该模块的基本流程，如何快速构建自己的模型并进行训练。

