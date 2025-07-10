import os

from PIL import Image

def merge(im1, im2, title, loc):
    images = [Image.open(x) for x in [im1,im2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(f"{loc}/"+title+" Combined.png")

if __name__ == "__main__":
    '''
    a = "whole_plots/blue-green/control_slow/no_train/Lr=0.0001 0.png"
    b = "whole_plots/scatter_plots/no_train/no_training.png"
    merge(a, b, title="no_train ", loc = "whole_plots/combined/control_slow")
   

    for i in range(50):
        a = f"whole_plots/blue-green/control_slow/unsup/Lr=0.007 {i}.png"
        b = "whole_plots/scatter_plots/unsup, every batch/"+f"unsup lr = 0.0001, 0 {i}.png"
        title = f"Lr=0.0001 {i}.png".split(" ")[1].strip(".png")
        merge(a, b,title="unsup "+title, loc = "whole_plots/combined/control_slow")

    '''
    for i in range(5):
        for j in range(5):

            a = f"whole_plots/blue-green/control_slow/sup/Lr=0.007 {i*5+j}.png"
            b = f"whole_plots/scatter_plots/sup_control_slow/sup lr = 0.0001, {i} {j}.png"
            title = str(i) + " "+ str(j)
            merge(a, b, title="z sup " + title, loc = "whole_plots/combined/control_slow")
