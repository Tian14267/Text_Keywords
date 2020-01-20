#coding: utf-8
##########################
##	@Author: FrankFan
##  @Data: 19-12-25
#########################
import os
import jieba
import json
import math
import numpy as np
import pandas as pd


class Tools(object):
	def __init__(self):
		super(Tools, self).__init__()

	def write_file(self,file, data):  ### 输入数组
		with open(file, 'w', encoding='utf-8') as f:
			f.write('\n'.join(data) + '\n')
		f.close()

	def read_file(self,file):
		with open(file, 'r', encoding='utf-8') as f:
			lines = f.readlines()
			new_lines = []
			for line in lines:
				new_line = line.strip().replace('\n', '')
				new_lines.append(new_line)
		f.close()
		return new_lines

	def jieba_data(self,train_data, stop_word_file, vocab_dir):
		##########  用来输出分词后的词汇  ##########################
		#### train_data: 训练数据
		#### stop_word_file：停用词列表
		#### vocab_dir：词典输出列表

		contents = []
		all_lines = self.read_file(train_data)
		for i, line in enumerate(all_lines):
			label, content = line.strip().split('\t')
			content_cut = jieba.cut(content, cut_all=False)
			for content_one in content_cut:
				if content_one not in contents:
					contents.append(content_one)
			print("line {:d} is ok !".format(i))
		print("contents len: ", len(contents))

		all_stop_words = self.read_file(stop_word_file)
		new_contents = []
		for content in contents:
			if content in all_stop_words:
				pass
			else:
				new_contents.append(content)
		print("New contents len: ", len(new_contents))
		self.write_file(vocab_dir, new_contents)
		return new_contents

	def data_smooth(self,data, stop_word_dir=None):
		###  1: 去除数组中的非中文字符。包括标点符号，英文字母，特殊符号等
		###  2: 去除停用词
		###  3：去除单字
		### data: 一个一维数组:['去除','中的','括标',...]
		import re
		zhmodel = re.compile(u'[\u4e00-\u9fa5]')  # 检查中文
		all_stop_words = []
		if stop_word_dir is not None:
			all_stop_words = self.read_file(stop_word_dir)  ## 加载停用词
		new_data = []
		for one_data in data:
			match = zhmodel.search(one_data)
			if match and one_data not in all_stop_words:  ###去除非中文和停用词
				if len(one_data) > 1:  ### 去除单字
					new_data.append(one_data)
		return new_data

	def Mix_algorithm(self,method_num,TF_result,TR_result):
		################
		## 参数：1：融合算法； 2：TF-IDF提取的keywords;  3:TextRank提取的keywords
		## 融合算法："Normal"，"Sum"，"Weight_Cross"
		if method_num is "Normal":
			### 法一：归一化
			########## 计算TF归一化
			result_sum = np.sum(np.array(list(TF_result.values())))
			keys = []
			for key in TF_result.keys():
				TF_result[key] = TF_result[key] / result_sum
				keys.append(key)
			########## 计算TR归一化
			result_tr_sum = np.sum(np.array(list(TR_result.values())))
			for key in TR_result.keys():
				TR_result[key] = (TR_result[key]) / result_tr_sum
				keys.append(key)

			keys = list(set(keys))
			mix_result = {}
			for key in keys:
				if key not in TF_result.keys():
					TF_result[key] = 0
				if key not in TR_result.keys():
					TR_result[key] = 0
				mix_result[key] = TF_result[key] + TR_result[key]
			res = sorted(mix_result.items(), key=lambda x: x[1], reverse=True)  ## 排序
			#print("Normal 融合后：", res)
		elif method_num is "Sum":
			### 法二：直接权值
			mix_result_2 = {}
			keys_tf = [key for key in TF_result.keys()]
			keys_tr = [key for key in TR_result.keys()]
			keys = keys_tf + keys_tr
			for key in keys:
				if key not in TF_result.keys():
					TF_result[key] = 0
				if key not in TR_result.keys():
					TR_result[key] = 0
				mix_result_2[key] = TF_result[key] * 0.1 + TR_result[key]
			res = sorted(mix_result_2.items(), key=lambda x: x[1], reverse=True)  ## 排序
			#print("Sum 融合后：", res)
		elif method_num is "Weight_Cross":
			### 法三：权值交叉
			########## 计算TF归一化
			result_sum = np.sum(np.array(list(TF_result.values())))
			keys = []
			TF_Normal = {}
			TR_Normal = {}
			for key in TF_result.keys():
				TF_Normal[key] = TF_result[key] / result_sum
				keys.append(key)
			########## 计算TR归一化
			result_tr_sum = np.sum(np.array(list(TR_result.values())))
			for key in TR_result.keys():
				TR_Normal[key] = (TR_result[key]) / result_tr_sum
				keys.append(key)
			keys = list(set(keys))
			mix_result = {}
			K = 0.1  ### 调整基数
			for key in keys:
				if key not in TF_result.keys():
					TF_result[key] = 0
					TF_Normal[key] = 0
				if key not in TR_result.keys():
					TR_result[key] = 0
					TR_Normal[key] = 0
				mix_result[key] = TR_Normal[key] * TF_result[key] * K + TF_Normal[key] * TR_result[key]
			res = sorted(mix_result.items(), key=lambda x: x[1], reverse=True)  ## 排序
			#print("Weight_Cross 融合后：", res)

	def IDF(self,file_lines):
		### 输入：新闻内容   type: list
		print("#########  计算文本IDF  Begin #############")
		num_file = len(file_lines)  ### 新闻总数
		num_temp = int(num_file/3)
		print("### Block Num: ",num_temp)
		dic_idf = {}
		for num_i in range(3): ### 0,1,2
			content_file = []
			for i, line in enumerate(file_lines[num_temp*num_i:(num_temp*(num_i+1)-1)]):
				content_cut = jieba.cut(line, cut_all=False)
				content_cut_smooth = self.data_smooth(content_cut, None)  ## 数据平滑，只保留汉字
				content_file.append(content_cut_smooth)
				if i % 200000 == 0:
					print("## 已分词新闻数：",i)
			print("### 数据分词和平滑结束 ###")
			dic_i = {}
			for c_i,content_news in enumerate(content_file):  ### 遍历每一个新闻
				# print("去重前：",len(content_news))
				content_news = list(set(content_news))  ### 关键词去重
				#print("去重后：", len(content_news))
				for content in content_news: ### 遍历新闻中每个词
					if content in dic_i.keys():  ### 统计包含该词的文档数
						dic_i[content] = dic_i[content] + 1
					else:
						dic_i[content] = 1
				if c_i % 200000 == 0:
					print("## 已处理新闻数：",c_i)
			print("############  处理结束！  ###############")

			for key in dic_i.keys():
				if key in dic_idf.keys():
					dic_idf[key] = dic_idf[key] + dic_i[key]
				else:
					dic_idf[key] = dic_i[key]
			dic_i.clear()
		print(dic_idf)


		##########  计算 IDF   ##########################
		print("开始计算IDF...")
		final_idf = {}
		for key in dic_idf.keys():
			final_idf[key] = math.log(num_file / (dic_idf[key] + 1))

		print("IDF关键词总数：",len(final_idf.keys()))
		# final_idf["File_Num"] = num_file
		##### 输出 IDF 文件
		with open("dict_idf_temp.json", "w", encoding="utf-8") as f:
			json.dump(final_idf, f)
		print("IDF制作完成！")
		#return final_idf

def temp():
	with open("dict_article_num.json", 'r', encoding='utf-8') as f:
		words = json.load(f)
	with open("dict_idf.json", 'r', encoding='utf-8') as f_idf:
		words_idf = json.load(f_idf)
	#print(words["Total_num"])
	#print(words)
	print(words_idf)
	#print(len(words))

def read_news(path):
	files_name = os.listdir(path)  # 采用listdir来读取所有文件
	file_full_name = [os.path.join(path,file) for file in files_name]
	file_contents = []
	i=0
	file_failed = [] ### 统计读取失败的文件
	for file in file_full_name:
		try:
			with open(file, 'r', encoding='utf-8') as f:
				lines = f.readlines()
				for line in lines:
					new_line = line.strip().replace('\n', '')
					file_contents.append(new_line)
			f.close()
			i = i+1
		except:
			file_failed.append(file)
		print("Deal file num: ",i)
	print("读取失败的文件：",file_failed)
	return file_contents




if __name__=='__main__':
	#####  制作 IDF 字典文件
	tools = Tools()
	file_contents = read_news("/home/hj/smbshare/fffan/Data/Sohu_News/")
	print("新闻总数：",len(file_contents))
	tools.IDF(file_contents)

	print("Done !")
