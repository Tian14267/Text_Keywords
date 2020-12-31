# Text_Keywords
## 功能实现
* 完成对文本的关键词提取。所用算法：基于`TF-IDF` 与 `TextRank` 融合算法 <br>
* 融合算法主要包括三种，分别是：归一化、权值相加以及权值交叉。可以自行选择和设置。<br>
* `Keywords_main_MutilSentence.py` 为读取所有文件并生成关键词；`Keywords_main_SingleSentence.py` 为读取单个文本内容并提取出关键词。`Keywords_Algorithm.py` 为所用到的算法；

## 注意
* IDF算法需要提前根据自己的数据集制作IDF文件。可以通过运行`keywords_tools.py`文件进行制作，但需要修改为自己的数据
