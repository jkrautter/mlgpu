import json
from random import shuffle
import numpy as np

def read_labeled_file(filename, flat=True, normalize=False, letters=False, shuffled=False, channels=False):
	result = {'labels': list(), 'images': list()}
	f = open(filename)
	for line in f:
		splitline = line.split("|")
		if not letters:
			result['labels'].append(int(splitline[0]))
		else:
			label = ord(splitline[0])
			if label > 97:
				result['labels'].append(label - 97 + 26)
			else:
				result['labels'].append(label - 65)
		image = json.loads(splitline[1])
		newimage = np.zeros(30*30)
		if channels:
			newimage = np.zeros([30, 30, 1])
		for i in range(30):
			for j in range(30):
				if normalize:
					image[i][j] = float(image[i][j]) / 255.0
				if channels:
					newimage[i][j][0] = image[i][j]
				elif flat:
					newimage[i*30 + j] = image[i][j]
		if flat or channels:
			result['images'].append(newimage)
		else:
			result['images'].append(image)
	f.close()
	if shuffled:
		combined = list(zip(result['images'], result['labels']))
		shuffle(combined)
		result['images'], result['labels'] = zip(*combined)
	return result

def shuffle_data(data):
	combined = list(zip(data['images'], data['labels']))
	shuffle(combined)
	data['images'], data['labels'] = zip(*combined)
	return data

def read_unlabeled_file(filename, flat=True, normalize=False, channels=False, shuffled=False):
	result = list()
	f = open(filename)
	for line in f:
		image = json.loads(line)
		newimage = np.zeros(30*30)
		if channels:
			newimage = np.zeros([30, 30, 1])
		for i in range(30):
			for j in range(30):
				if normalize:
					image[i][j] = float(image[i][j]) / 255.0
				if channels:
					newimage[i][j][0] = image[i][j]
				elif flat:
					newimage[i*30 + j] = image[i][j]
		if flat or channels:
			result.append(newimage)
		else:
			result.append(image)
	f.close()
	if shuffled:
		shuffle(result)
	return result
