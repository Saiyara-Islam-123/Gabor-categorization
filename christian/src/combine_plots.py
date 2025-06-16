import os

from PIL import Image

def merge(im1, im2, title):
    images = [Image.open(x) for x in [im1,im2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save("whole_plots/combined/slow_lr/"+title+" Combined.png")

if __name__ == "__main__":

    '''
    a = "whole_plots/blue-green/slow_lr/no_train/Lr=0.0001 0.png"
    b = "whole_plots/scatter_plots/no_train/no_training.png"
    merge(a, b, title="no_train ")


    '''
    lines = os.listdir("whole_plots/blue-green/slow_lr/sup")

    scatter_plots = os.listdir("whole_plots/scatter_plots/sup, every batch, slow lr")

    for i in range(50):
        a = "whole_plots/blue-green/slow_lr/sup/" + lines[i]
        b = "whole_plots/scatter_plots/sup, every batch, slow lr, 1 epoch/"+scatter_plots[i]
        title = lines[i].split(" ")[1].strip(".png")
        merge(a, b,title="z sup "+title)
    
