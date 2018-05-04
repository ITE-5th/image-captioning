import matplotlib.pyplot as plt
import skimage.transform
from PIL import Image


def attention_visualization(image_name, caption, alphas):
    image = Image.open(image_name)
    image = image.resize([224, 224], Image.LANCZOS)
    plt.subplot(4, 5, 1)
    plt.imshow(image)
    plt.axis('off')

    words = caption
    for t in range(len(words)):
        if t > 18:
            break
        plt.subplot(4, 5, t + 2)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        # print alphas
        alp_curr = alphas[t, :].view(14, 14)
        alp_img = skimage.transform.pyramid_expand(alp_curr.numpy(), upscale=16, sigma=20)
        plt.imshow(alp_img, alpha=0.85)
        plt.axis('off')
    plt.show()