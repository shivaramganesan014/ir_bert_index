import sys
import bert
import json


def get_item(indices, resultScores):

	index_lookup = "index_lookup.json"
	contents = []
	result = []
	with open(index_lookup, "r+") as f:
		contents = json.load(f)
	if len(contents) > 0:
		for i in indices:
			if(i < len(contents)):
				m = {}
				m["url"] = contents[i]
				m["score"] = resultScores[i]
				result.append(m)
	return result


query = "sample query"
print(len(sys.argv))
if(len(sys.argv)>=2):
	query = sys.argv[1]	
resultIndex, resultScores = bert.query_index(query)
if len(resultIndex) == 1 and resultIndex[0] == -1:
	print("No results found")
else:
	get_item(resultIndex, resultScores)




