import json

with open('emoji_images_with_word2vec.json') as data_file:    
    data = json.load(data_file)

with open('emoji_images_with_word2vec_medium_small.json', 'w') as fp:
    json.dump(data[:1212], fp, indent=4)