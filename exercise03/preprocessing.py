import json

def read_labeled_file(filename, flat=True):
	result = {'labels': list(), 'images': list()}
	f = open(filename)
	for line in f:
		splitline = line.split("|")
		result['labels'].append(splitline[0])
		image = json.loads(splitline[1])
		flatimage = list()
		for i in range(30):
			for j in range(30):
				flatimage.append(image[i][j])
		if flat:
			result['images'].append(flatimage)
		else:
			result['images'].append(image)
	f.close()
	return result
def read_unlabeled_file(filename, flat=True):
	result = list()
	f = open(filename)
	for line in f:
		image = json.loads(line)
		flatimage = list()
		for i in range(30):
			for j in range(30):
				flatimage.append(image[i][j])
		if flat:
			result.append(flatimage)
		else:
			result.append(image)
	f.close()
	return result
