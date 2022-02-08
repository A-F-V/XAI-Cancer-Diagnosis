import io
from PIL import Image
import mlflow


def log_plot(plt, name, logger=None, **kwargs):
    if logger == None:
        logger = mlflow
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    logger.log_image(image=im, artifact_file=f"{name}.png", **kwargs)
