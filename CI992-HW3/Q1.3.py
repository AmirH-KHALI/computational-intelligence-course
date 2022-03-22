# Q1.3_graded
# Do not change the above line.

from PIL import Image, ImageFont
import os
import numpy as np
import random

# Q1.3_graded
# Do not change the above line.

path = '/content/CI992/HW3/Q1.3'
chars = 'ABCDEFGHIJ'
fonts = [16, 32, 64]
noise_list = [0.1, 0.3, 0.6]

def generate_data():
    for f in fonts:
        font = ImageFont.truetype(f'{path}/arial.ttf', f)
        for ch in chars:
            im = Image.Image()._new(font.getmask(ch))
            im.save(f'{path}/data/{ch}_{str(f)}.bmp')


def get_noisy_image(img, noise):
    img_arr = np.array(img)

    pixels_num = img_arr.shape[0] * img_arr.shape[1]

    new_image = np.copy(img_arr)
    random_pixels = random.sample(range(0, pixels_num), 
                                  int(pixels_num * noise))

    for pixel in random_pixels:
        row = pixel // img_arr.shape[1]
        col = pixel % img_arr.shape[1]
        
        if img_arr[row][col] > 128:
            new_image[row][col] = 0
        else:
            new_image[row][col] = 255

    new_image = Image.fromarray(new_image, mode='L')
    return new_image


def generate_noisy_data():
    temp = 0
    for noise in noise_list:
        for file_name in os.listdir(f'{path}/data/'):
            if file_name.split('.')[1] != 'bmp':
                continue

            img = Image.open(f'{path}/data/{file_name}', mode='r')
            
            new_image = get_noisy_image(img, noise)
            
            noise_p = int(noise * 100)
            new_image.save(f'{path}/noisy_data/{str(noise_p)}_{file_name}')


generate_data()
generate_noisy_data()

# Q1.3_graded
# Do not change the above line.

class HopfieldNet:

    def __init__(self, n):
        self.n = n
        self.weights = np.zeros((n, n))

    def train(self, input):
        
        p = np.dot(input.T, input)
        
        np.fill_diagonal(p, 0)
        
        self.weights += p
    
    def restore(self, x):
        p = np.sum(x * self.weights, axis=1)
        return np.array(np.sign(p.T))

    def update(self, input):
        current_x = input
        last_x = np.copy(current_x)
        for i in range(1000):
            current_x = self.restore(current_x)
            if np.array_equal(current_x, last_x):
                break
            last_x = np.copy(current_x)
        return current_x


# Q1.3_graded
# Do not change the above line.

path = '/content/CI992/HW3/Q1.3'
dim = 40
chars = 'ABCDEFGHIJ'
fonts = [16, 32, 64]
noise_list = [10, 30, 60]

def generate_patterns(font_size):

    patterns = []
    for file_name in os.listdir(f'{path}/data/'):
        if file_name.split('.')[1] != 'bmp' or file_name.split('.')[0].split('_')[1] != str(font_size):
                continue
        
        img = Image.open(f'{path}/data/{file_name}', mode='r')
        img = img.resize((dim, dim))
        img_arr = np.asarray(img, dtype=np.uint8)

        pattern = np.ones(img_arr.shape)
        pattern[img_arr <= 60] = -1

        patterns.append(pattern.flatten())

    return np.array(patterns)


for f in fonts:
    patterns = generate_patterns(f)
    
    hn = HopfieldNet(dim ** 2)
    hn.train(patterns)
    
    for noise in noise_list:
        for ch in chars:
            img = Image.open(f'{path}/noisy_data/{str(noise)}_{ch}_{str(f)}.bmp', mode='r')
            img = img.resize((dim, dim))
            img_arr = np.asarray(img, dtype=np.uint8)

            pattern = np.ones(img_arr.shape)
            pattern[img_arr <= 60] = -1

            result_pattern = hn.update(pattern.flatten())

            result_arr = np.where(result_pattern >= 0, 255, 0)
            result_arr = np.array(result_arr, dtype=np.uint8)

            out = Image.fromarray(result_arr.reshape((dim, dim)), mode='L')
            out.save(f'{path}/result/{str(noise)}_{ch}_{str(f)}.bmp')

            org_img = Image.open(f'{path}/data/{ch}_{str(f)}.bmp', mode='r')
            org_img = org_img.resize((dim, dim))
            org_img_arr = np.asarray(org_img, dtype=np.uint8)

            org_img_pattern = np.ones(org_img_arr.shape)
            org_img_pattern[org_img_arr <= 60] = -1
            org_img_pattern = org_img_pattern.flatten()

            rate = 100 - 100 * np.sum(np.abs(result_pattern - org_img_pattern)) / (2 * dim * dim)
            print(f'Accuracy for noise: {str(noise)}, character: {ch}, font: {str(f)} --> {str(rate)}')

