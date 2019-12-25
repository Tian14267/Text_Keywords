#coding: utf-8
##########################
##	@Author: FengfeiFan
##  @Data: 19-12-25
#########################
import numpy as np
import jieba
import json
import keywords_tools as KT

class TFIDF(object):
	def __init__(self,tools):
		self.tools = tools  # 加载工具包
		#self.file_lines = self.tools.read_file(file)
		with open("dict_idf_temp.json", 'r', encoding='utf-8') as f_idf:
			self.words_idf = json.load(f_idf)

	def Do_keywords(self,file):
		content_file = []
		self.file_lines = [file]
		for i, line in enumerate(self.file_lines):
			content_cut = jieba.cut(line, cut_all=False)
			content_cut_smooth = self.tools.data_smooth(content_cut, None)  ## 数据平滑，只保留汉字
			content_file = content_cut_smooth

		dic_tf = {}  ### 统计词频
		content_num = 0  ### 初始化
		for content_num, content in enumerate(content_file):
			if content in dic_tf.keys():
				dic_tf[content] = dic_tf[content] + 1
			else:
				dic_tf[content] = 1
		for key in dic_tf.keys():
			assert content_num != 0
			dic_tf[key] = dic_tf[key] / content_num

		#####  计算词汇的 TF-IDF
		dic_tfidf = {}
		for key in dic_tf.keys():
			if key in self.words_idf.keys():
				tfidf = dic_tf[key] * self.words_idf[key]
			else:
				tfidf = 0
			dic_tfidf[key] = tfidf
		#dic_tfidf_2 = sorted(dic_tfidf)
		dic_tfidf_2 = sorted(dic_tfidf.items(), key=lambda x: x[1], reverse=True)
		#print(dic_tfidf) ### 原始关键词
		#print(dic_tfidf_2)  ### 排序后的关键词
		for ii, key in enumerate(dic_tfidf_2): ###排序后关键词与得分
			#print(key, ":", dic_tfidf[key])
			if ii > 10:
				break
		return dic_tfidf,dic_tfidf_2

class TextRank(object):
	def __init__(self, window, alpha, iternum,tools):
		super(TextRank, self).__init__()
		self.window = window
		self.alpha = alpha
		self.iternum = iternum #迭代次数
		self.tools = tools  # 加载工具包

	# 对句子进行分词
	'''
	def cutSentence(self):
		#jieba.load_userdict('user_dict.txt')
		tag_filter = ['a','d','n','v']
		seg_result = pseg.cut(self.sentence)
		self.word_list = [s.word for s in seg_result if s.flag in tag_filter]
		print(self.word_list)
	'''

	def calculate_textrank(self,sentence):
		self.sentence = sentence
		seg_result = jieba.cut(self.sentence, cut_all=False)
		self.word_list = self.tools.data_smooth(data=seg_result,stop_word_dir="./data/stop_word.txt")
		#self.word_list = [word for word in seg_result]
		#print(self.word_list)

		#createNodes: 根据窗口，构建每个节点的相邻节点,返回边的集合
		tmp_list = []
		word_list_len = len(self.word_list)
		self.edge_dict = {} #记录节点的边连接字典
		for index, word in enumerate(self.word_list):
			if word not in self.edge_dict.keys():
				tmp_list.append(word)
				tmp_set = set()
				left = index - self.window + 1#窗口左边界
				right = index + self.window#窗口右边界
				if left < 0: left = 0
				if right >= word_list_len: right = word_list_len
				for i in range(left, right):
					if i == index:
						continue
					tmp_set.add(self.word_list[i])
				self.edge_dict[word] = tmp_set

		#createMatrix: 根据边的相连关系，构建矩阵
		self.matrix = np.zeros([len(set(self.word_list)), len(set(self.word_list))])
		self.word_index = {}#记录词的index
		self.index_dict = {}#记录节点index对应的词

		for i, v in enumerate(set(self.word_list)):
			self.word_index[v] = i
			self.index_dict[i] = v
		for key in self.edge_dict.keys():
			for w in self.edge_dict[key]:
				self.matrix[self.word_index[key]][self.word_index[w]] = 1
				self.matrix[self.word_index[w]][self.word_index[key]] = 1
		#归一化
		for j in range(self.matrix.shape[1]):
			sum = 0
			for i in range(self.matrix.shape[0]):
				sum += self.matrix[i][j]
			for i in range(self.matrix.shape[0]):
				self.matrix[i][j] /= sum

		#calPR: 根据textrank公式计算权重
		self.PR = np.ones([len(set(self.word_list)), 1])
		for i in range(self.iternum):
			self.PR = (1 - self.alpha) + self.alpha * np.dot(self.matrix, self.PR)

		#printResult: 输出词和相应的权重
		word_pr = {}
		for i in range(len(self.PR)):
			word_pr[self.index_dict[i]] = self.PR[i][0]### 关键词及权重
		####  计算分值
		value_price = 0
		for value in word_pr.values():
			value_price = value_price + value
		for key in word_pr.keys():
			word_pr[key] = word_pr[key]/value_price
		res = sorted(word_pr.items(), key=lambda x: x[1], reverse=True) ## 排序
		#res = sorted(word_pr, reverse=False)
		#print(word_pr)
		return word_pr,res

if __name__ == '__main__':
	s = '程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。'
	tools = KT.Tools()
	tr = TextRank(s, 3, 0.85, 700,tools)
	tr.calculate_textrank()




