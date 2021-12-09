from bs4 import BeautifulSoup as bs
from PIL import ImagePath, Image, ImageDraw
import numpy as np
"""Creates a mask for the segmentation of the H&E image
"""


class SegmentationMask:  # for MoNuSeg
    def __init__(self, annotation_path):
        """Constructs a mask for the segmentation of the H&E image

        Args:
            annotation_path (str): path to the annotation file
        """
        self.annotation_path = annotation_path

    def create_mask(self):  # todo memoise
        """Creates a mask for the segmentation of the H&E image

        Returns:
            numpy.ndarray: mask
        """
        content = ""
        with open(self.annotation_path) as f:
            content = "\n".join(f.readlines())
        soup = bs(content, "xml")
        annotations = soup.Annotations.find_all("Annotation")

        sx, sy = 0, 0  # size of mask bounding box
        paths = []
        for annotation in annotations:
            for region in annotation.Regions.find_all("Region"):
                vertices = region.Vertices.find_all("Vertex")
                points = list(map(lambda v: (float(v['X']), float(v['Y'])), vertices))

                ipath = ImagePath.Path(points)
                paths.append(ipath)

                bounding_box = list(map(int, map(np.ceil, ipath.getbbox())))  # bound
                sx, sy = max(sx, bounding_box[2]), max(sy, bounding_box[3])

        img = Image.new("RGB", (sx, sy), color="#000000")
        drawing = ImageDraw.Draw(img)
        for path in paths:
            drawing.line(path, fill="white")
        return img

# todo: fill in shapes, make size of mask equal to original image
