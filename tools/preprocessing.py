from PIL import Image

def resize(image, size = (224, 224)):
    image = image.resize(size)
    return image

