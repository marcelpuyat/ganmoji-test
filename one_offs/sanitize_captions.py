

import json
import re
from pprint import pprint

with open('emoji_images_high_quality.json') as data_file:    
    data = json.load(data_file)

for pair in data:
	pair['title'] = re.sub('[^A-Za-z0-9\- &\*#]+', '', pair['title']).strip()

with open('sanitized_emoji_images_high_quality.json', 'w') as fp:
    json.dump(data, fp, indent=4)