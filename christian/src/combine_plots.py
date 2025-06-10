from PIL import Image

def merge(im1, im2):
    images = [Image.open(x) for x in [im1, im2]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save("Combined " +im1.strip(".png") + ".png")

for i in range(8):
    for j in range(5):
        merge(f"sup_net_weights_{i} {j}.png", f"sup {i} {j} .png")