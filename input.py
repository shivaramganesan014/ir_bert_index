import os
import json
import bert

def add_to_file(filename, content_list):
	infile = open(filename, 'w+')
	try:
		content_list.extend(json.load(infile))
	except:
		pass
		# print("empty file")
	json.dump(content_list, infile)
	# with open(filename, 'r') as infile:
	# 	try:
	# 		result.extend(json.load(infile))
	# 	except:
	# 		print("error reading ", filename)
	# with open(filename, 'w') as output_file:
		

def split(sentence):
	split_param = 500
	chunks = [sentence[i:i+split_param] for i in range(0, split_param, split_param)]
	return chunks


path = "/home/cs242/project/data/ir_combined_data/"
urls = []
index_lookup = "index_lookup.json"
files = os.listdir(path)
with open(index_lookup ,'w') as f:
	pass
for file_ in files:
	if("sample" in file_):
		continue
	print("keys for "+ file_)
	batch_size = 100
	with open (path+file_, "r") as f:
		sentences = []
		contentjson = json.load(f)
		index = 0
		for jsonC in contentjson:
			content = ""
			if "title" in jsonC:
				content+=jsonC["title"]
			if "body" in jsonC:
				content+=jsonC["body"]
			if(content != ""):
				split_contents = split(content)
				for c in split_contents:
					sentences.append(c)
					urls.append({index: jsonC["url"]})
					index+=1
			if(len(sentences)>batch_size):
				break
		add_to_file(index_lookup, urls)
		bert.index_sentences(sentences)

	break	# print(contentjson.keys())

