import json
from PIL import Image
import numpy as np
np.set_printoptions(threshold='nan')
import scipy.misc

def get_pixels_for_filename(filename):
    img = scipy.misc.imread(filename, mode='RGBA')
    img = Image.fromarray(img)
    return np.array(img.getdata())

def main():
    # with open('emoji_images.json') as data_file:
    #     index_data = json.load(data_file)
    # index_data an array of dicts where each dict represents an
    # emoji sample and is of the form:
    # {
    #   'filename': the filename containing the emoji image,
    #   'title': you can think of this as the emoji caption/label
    # }
    # pix = get_pixels_for_filename(index_data[654]['filename'])
    pix = get_pixels_for_filename('emoji1.png')
    # print(data[654]['title'])
    print(pix.shape)
    print(pix)

if __name__ == '__main__':
    main()