from bs4 import BeautifulSoup as bs
from PIL import ImagePath, Image, ImageDraw
import numpy as np
"""Creates a mask for the segmentation of the H&E image
"""


class SemanticSegmentationMask:  # for MoNuSeg
    def __init__(self, annotation_path):
        """Constructs a mask for the segmentation of the H&E image. Binary mask (either cell or not cell)

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

        img = Image.new("L", bb, color=0)
        drawing = ImageDraw.Draw(img)
        for path in paths:
            if filled:
                drawing.polygon(path, fill=255)
            else:
                ipath = ImagePath.Path(path)
                drawing.line(ipath, fill=255)
        return np.array(img)

# todo: test
