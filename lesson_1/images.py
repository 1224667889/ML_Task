import PIL.Image as Image
import os


def image_compose(label):
    image_names = [name for name in os.listdir(label + "/")]
    to_image = Image.new('RGB', (10 * 100, 10 * 100))
    for y in range(1, 10 + 1):
        for x in range(1, 10 + 1):
            from_image = Image.open(label + "/" + image_names[10 * (y - 1) + x - 1]).resize(
                (100, 100), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * 100, (y - 1) * 100))
    return to_image.save(label + ".jpg")


if __name__ == '__main__':
    image_compose("Cat")
    image_compose("Dog")
