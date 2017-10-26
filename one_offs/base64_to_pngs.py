import json

with open('/Users/marcelpuyat/Downloads/emojis2.json') as data_file:    
    data = json.load(data_file)

image_num = 1
for pair in data:
	base64 = pair.pop('base64', None)
	filename = './high_quality_data/emoji' + str(image_num) + '.png'
	fh = open(filename, "wb")
	fh.write(base64.decode('base64'))
	fh.close()
	pair['filename'] = filename
	image_num += 1
