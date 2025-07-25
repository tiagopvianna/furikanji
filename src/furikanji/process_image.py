import fire
from PIL import Image
from pathlib import Path
from loguru import logger

from src.furikanji.core_logic import CoreLogic


def process_single_image(
    image_path: str,
    output_path: str = "output.png",
    device: str = "cpu",
    force_cpu: bool = False,
):
    """
    Processes a single manga image to detect and overlay text.

    Args:
        image_path: Path to the input image.
        output_path: Path to save the output image with overlaid text.
        device: The device to use for processing ('cpu', 'cuda').
        force_cpu: If True, forces CPU usage.
    """
    if force_cpu:
        device = "cpu"

    mg = CoreLogic()

    print(image_path)
    mg.process_volume(image_path)

if __name__ == '__main__':
    fire.Fire(process_single_image)