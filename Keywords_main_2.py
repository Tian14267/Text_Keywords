#coding: utf-8
##########################
##	@Author: FengfeiFan
##  @Data: 19-12-25
#########################
import numpy as np
import time
import Keywords_Algorithm as KA
import keywords_tools as KT

def Pred_model():
	tools = KT.Tools()
	tf_idf = KA.TFIDF(tools)
	tr = KA.TextRank(3, 0.85, 700,tools)  ### 参数：句子，窗大小，alpha，迭代次数，工具包
	return tf_idf,tr,tools

def Mix_keywords(tf_idf,tr,tools,file):
	result,result_sort = tf_idf.Do_keywords(file) ### 第一个为原始字典，第二个为排序后的字典
	#print("TF_IDF result: ",result_sort)

	result_tr,result_tr_sort = tr.calculate_textrank(file) ### 第一个为原始字典，第二个为排序后的字典
	#print("TextRank result: ", result_tr_sort)

	#####  融合
	method_num = "Weight_Cross"  ### "Normal" ，"Sum"， "Weight_Cross"
	mix_result = tools.Mix_algorithm(method_num,result,result_tr) ## 参数：1：融合算法； 2：TF-IDF提取的keywords;  3:TextRank提取的keywords
	#print("Weight_Cross 融合后：", mix_result)

if __name__=='__main__':
	path = "./data/cnews.test.txt"
	start = time.time()
	with open(path,'r', encoding='utf-8') as f:
		lines = f.readlines()
		new_lines = []
		for line in lines:
			new_line = line.strip().replace('\n', '')
			new_lines.append(new_line)
	f.close()
	all_character = 0
	tf_idf,tr,tools = Pred_model()
	for i,one_line in enumerate(new_lines):
		all_character = all_character + len(one_line)
		#sentence = "国美澄清：黄光裕资产被冻结一案与公司无关国美电器的一份声明指出，目前没有监管机构或司法机关就香港法院的命令与国美进行过接洽。观点地产网讯：据外媒消息，国美电器控股有限公司8月7日表示，香港法院批准香港证监会的申请，下令冻结公司创始人黄光裕资产一案，与国美电器无关。国美电器的一份声明指出，目前没有监管机构或司法机关就香港法院的命令与国美进行过接洽。据悉，黄光裕、其妻杜鹃和两家控股公司处置或出售所持的16.6亿港元国美电器股份被禁止。黄光裕及其亲友持有国美电器34%的股份。 我要评论"
		Mix_keywords(tf_idf,tr,tools,one_line)
		print("完成：",i)
	end = time.time()
	print("总字数：",all_character)
	print("总时间：",(end-start))
	