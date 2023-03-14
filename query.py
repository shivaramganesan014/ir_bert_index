import sys
import bert
import json


def get_item(indices):

	index_lookup = "index_lookup.json"
	contents = []
	result = []
	with open(index_lookup, "r+") as f:
		contents = json.load(f)
	if len(contents) > 0:
		for i in indices:
			if(i < len(contents)):
				result.append(contents[i])
	print(result)
	return result


query = "Nemo is a fish"
print(len(sys.argv))
if(len(sys.argv)>=2):
	query = sys.argv[1]	
print("results for "+query)
resultIndex = bert.query_index(query)
print(len(resultIndex))
if len(resultIndex) == 1 and resultIndex[0] == -1:
	print("No results found")
else:
	get_item(resultIndex)




