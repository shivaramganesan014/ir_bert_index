import os
import json
import bert

path = "/home/cs242/project/data/ir_combined_data/"
files = os.listdir(path)
for file_ in files:
	if("sample" in file_):
		continue
	print("keys for "+ file_)
	with open (path+file_, "r") as f:
		sentences = []
		len = 0
		contentjson = json.load(f)
		for jsonC in contentjson:
			if "title" in jsonC:
				sentences.append(jsonC["title"])
				len+=1
			if "body" in jsonC:
				sentences.append(jsonC["body"])
		bert.index(sentences)
	break	# print(contentjson.keys())

