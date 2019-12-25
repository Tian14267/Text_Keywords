#coding: utf-8
##########################
##	@Author: FengfeiFan
##  @Data: 19-12-25
#########################
import numpy as np
import Keywords_Algorithm as KA
import keywords_tools as KT

def Mix_keywords(file):
	tools = KT.Tools()
	tf_idf = KA.TFIDF(file,tools)
	result,result_sort = tf_idf.Do_keywords() ### 第一个为原始字典，第二个为排序后的字典
	print("TF_IDF result: ",result_sort)

	tr = KA.TextRank(file, 3, 0.85, 700,tools)  ### 参数：句子，窗大小，alpha，迭代次数，工具包
	result_tr,result_tr_sort = tr.calculate_textrank() ### 第一个为原始字典，第二个为排序后的字典
	print("TextRank result: ", result_tr_sort)

	#####  融合
	method_num = "Weight_Cross"  ### "Normal" ，"Sum"， "Weight_Cross"
	tools.Mix_algorithm(method_num,result,result_tr) ## 参数：1：融合算法； 2：TF-IDF提取的keywords;  3:TextRank提取的keywords


if __name__=='__main__':
	#sentence = "黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂，此前的比赛中，双方战成2-2平，因此本场比赛对于两支球队来说都非常重要，赛前双方也公布了首发阵容：湖人队：费舍尔、科比、阿泰斯特、加索尔、拜纳姆黄蜂队：保罗、贝里内利、阿里扎、兰德里、奥卡福[新浪NBA官方微博][新浪NBA湖人新闻动态微博][新浪NBA专题][黄蜂vs湖人图文直播室](新浪体育)"
	
	Mix_keywords(sentence)
