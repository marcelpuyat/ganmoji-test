import json

with open('sanitized_emoji_images_high_quality.json') as data_file:    
    data = json.load(data_file)

with open('sanitized_emoji_images_high_quality_medium.json', 'w') as fp:
    json.dump(data[:1212], fp, indent=4)