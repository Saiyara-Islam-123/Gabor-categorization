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
    a = "whole_plots/blue-green/control/no_train_2_epochs/Lr=0.007 0.png"
    b = "whole_plots/scatter_plots/no_train/no_training.png"
    merge(a, b, title="no_train ", loc = "whole_plots/combined/control")


    lines = os.listdir("whole_plots/blue-green/control/unsup_2_epochs")

    scatter_plots = os.listdir("whole_plots/scatter_plots/unsup, every batch")

    for i in range(50):
        a = "whole_plots/blue-green/control/unsup_2_epochs/" + lines[i]
        b = "whole_plots/scatter_plots/unsup, every batch/"+scatter_plots[i]
        title = lines[i].split(" ")[1].strip(".png")
        merge(a, b,title="unsup "+title, loc = "whole_plots/combined/control")
    
    '''


    for i in range(2):
        for j in range(16):
            scatter_plot = f"sup lr = 0.007, {i} {j}.png"
            a = f"whole_plots/blue-green/control/sup_2_epochs/Lr=0.007 {j+16*i}.png"
            b = "whole_plots/scatter_plots/sup_control/" + scatter_plot
            title = str(i) + " "+ str(j)
            merge(a, b, title="z sup " + title, loc = "whole_plots/combined/control")
    

