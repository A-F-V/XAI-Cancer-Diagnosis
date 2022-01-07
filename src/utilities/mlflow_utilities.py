import io
from PIL import Image
import mlflow


def log_plot(plt, name):
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    mlflow.log_image(im, f"{name}.png")
