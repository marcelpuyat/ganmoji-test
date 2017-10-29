import json

with open('sanitized_emoji_images_high_quality.json') as data_file:    
    data = json.load(data_file)

with open('sanitized_emoji_images_high_quality_medium_large.json', 'w') as fp:
    json.dump(data[:6000], fp, indent=4)
