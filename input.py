import os
import json
import bert
import time
import faiss
import nltk
from nltk.corpus import stopwords

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

def remove_stopwords(text_tokens):
	tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
	return " ".join(tokens_without_sw)



path = "/home/cs242/project/data/ir_combined_data/"
urls = []
index_lookup = "index_lookup.json"
files = os.listdir(path)
with open(index_lookup ,'w') as f:
	pass

index = faiss.IndexFlatIP(768)
nltk.download('stopwords')
for file_ in files:
	# if("sample" in file_ or "wiki.jsonl" in file_ or "wikipedia.jsonl" in file_):
	# 	print("skipping "+file_)
	# 	continue
	if("sample" in file_):
		print("skipping "+file_)
		continue
	print("Reading "+ file_)
	batch_size = 2000
	with open (path+file_, "r") as f:
		sentences = []
		contentjson = json.load(f)
		c_index = 0
		for jsonC in contentjson:
			content = ""
			url = ""
			if "title" in jsonC:
				content+=jsonC["title"]
			if "body" in jsonC:
				content+=jsonC["body"]
			if "url" in jsonC:
				url = jsonC["url"]
			if("wiki.jsonl" in file_ or "wikipedia.jsonl" in file_):
				for key, value in jsonC.items():
					if ("Title" == key):
						content += value[key]
						url = "https://en.wikipedia.org/wiki/"+str(value["Title"])
					if("Content" in value):
						content += str(value["Content"])
			if("intro_text" in jsonC):
				content+= jsonC["intro_text"]
			if("wiki" not in url and "reddit" not in url and "stack" not in url):
				continue
			if(content != ""):
				content = remove_stopwords(content.split(" "))
				split_contents = split(content)
				for c in split_contents:
					sentences.append(c)
					urls.append({c_index: url})
					c_index+=1
			if(len(sentences)>batch_size):
				break
		add_to_file(index_lookup, urls)
		values = bert.index_sentences(index, sentences)
		if values is not None:
			for mean_pooled in values:
				if mean_pooled is not None:
					index.add(mean_pooled)
faiss.write_index(index,"sample_code.index")

