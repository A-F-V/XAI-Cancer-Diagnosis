from bs4 import BeautifulSoup as bs
from PIL import ImagePath, Image, ImageDraw
import numpy as np
"""Creates a mask for the segmentation of the H&E image
"""


class InstanceSegmentationMask:  # for MoNuSeg #todo COMPARE WITH SEMANTIC SEGMENTATION MASK and REFACTOR
    def __init__(self, annotation_path):
        """Constructs an instance mask for the segmentation of the H&E image. Each cell is labelled with a different number

        Args:
            annotation_path (str): path to the annotation file
        """
        self.annotation_path = annotation_path

    def create_mask(self, size=None, filled=True):  # todo memoise
        """Creates a mask for the segmentation of the H&E image

        Returns:
            numpy.ndarray: mask
        """
        content = ""
        with open(self.annotation_path) as f:
            content = "\n".join(f.readlines())
        soup = bs(content, "xml")
        annotations = soup.Annotations.find_all("Annotation")

        bb = (0, 0) if size == None else size  # size of mask bounding box
        paths = []
        for annotation in annotations:
            for region in annotation.Regions.find_all("Region"):
                vertices = region.Vertices.find_all("Vertex")
                points = list(map(lambda v: (float(v['X']), float(v['Y'])), vertices))

                ipath = ImagePath.Path(points)
                paths.append(points)

                bounding_box = list(map(int, map(np.ceil, ipath.getbbox())))  # bound
                if size == None:
                    bb = max(size[0], bounding_box[2]), max(size[1], bounding_box[3])

        img = Image.new("I", bb, color=0)
        drawing = ImageDraw.Draw(img)
        next_colour = 1
        for path in paths:
            if filled:
                drawing.polygon(path, fill=next_colour)
            else:
                ipath = ImagePath.Path(path)
                drawing.line(ipath, fill=next_colour)
            next_colour += 1
        return np.array(img)

# todo: test
