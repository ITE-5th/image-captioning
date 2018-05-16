import math

import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image


def attention_visualization(image_name, caption, alphas, regions=49):
    image = Image.open(image_name)
    image = image.resize([224, 224], Image.LANCZOS)
    plt.subplot(4, 5, 1)
    plt.imshow(image)
    plt.axis('off')
    words = caption
    regions = int(math.sqrt(regions))
    upscale = 224 / regions
    for t in range(len(words)):
        if t > 18:
            break
        plt.subplot(4, 5, t + 2)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        # print alphas
        alp_curr = alphas[t].view(regions, regions)
        alp_img = skimage.transform.pyramid_expand(alp_curr.detach().numpy(), upscale=upscale, sigma=20)
        plt.imshow(alp_img, alpha=0.7)
        plt.axis('off')
    plt.savefig(f'{image_name.replace(".jpg", "")}-attention.png')
    plt.show()
