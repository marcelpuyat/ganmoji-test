from gensim.models import KeyedVectors
import json
import re
import numpy as np

model = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

with open('sanitized_emoji_images.json') as data_file:    
    data = json.load(data_file)

no_valid_words = ['Curaao', 'keycap 10', 'keycap *', 'keycap #', 'kaaba', 'doughnut', 'selfie', 'merperson']

in_model = 0

data_with_vectors = []

for pair in data:
	title = pair['title']
	if title in model:
		pair['vector'] = model[title].tolist()
		data_with_vectors.append(pair)
	else:
		cc = '_'.join(title.split(' '))
		if cc in model:
			pair['vector'] = model[cc].tolist()
			data_with_vectors.append(pair)
		else:
			replaced = cc.replace('-', '_')
			if replaced in model:
				pair['vector'] = model[replaced].tolist()
				data_with_vectors.append(pair)
			else:
				stripped = re.sub('[^A-Za-z0-9]+', ' ', cc)
				words = stripped.split(' ')
				words_found = 0
				total = np.zeros(300)
				for word in words:
					if word in model:
						words_found += 1
						total = np.add(model[word], total)
				if words_found == 0:
					continue

				pair['vector'] = np.divide(total, words_found).tolist()
				data_with_vectors.append(pair)

with open('emoji_images_with_word2vec.json', 'w') as fp:
    json.dump(data_with_vectors, fp, indent=4)