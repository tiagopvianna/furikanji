import fire

from src.furikanji.core_logic import CoreLogic


def process_single_image(
    image_path: str,
    output_path: str = "output.png",
    device: str = "cpu",
    force_cpu: bool = False,
):
    if force_cpu:
        device = "cpu"

    mg = CoreLogic()

    print(image_path)
    mg.process_volume(image_path)


if __name__ == "__main__":
    fire.Fire(process_single_image)
