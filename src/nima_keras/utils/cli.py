import argparse
from path import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NIMA(Inception ResNet v2)")
    parser.add_argument(
        "-dir",
        type=str,
        default=None,
        help="Pass a directory to evaluate the images in it",
    )

    parser.add_argument(
        "-img",
        type=str,
        default=[None],
        nargs="+",
        help="Pass one or more image paths to evaluate them",
    )

    parser.add_argument(
        "-resize",
        type=str,
        default="false",
        help="Resize images to 224x224 before scoring",
    )

    parser.add_argument(
        "-rank",
        type=str,
        default="true",
        help="Whether to tank the images after they have been scored",
    )

    args = parser.parse_args()

    # give priority to directory
    if args.dir is not None:
        print("Loading images from directory : ", args.dir)
        imgs = Path(args.dir).files("*.png")
        imgs += Path(args.dir).files("*.jpg")
        imgs += Path(args.dir).files("*.jpeg")

    elif args.img[0] is not None:
        print("Loading images from path(s) : ", args.img)
        imgs = args.img

    else:
        raise RuntimeError("Either -dir or -img arguments must be passed as argument")

    resize_images = args.resize.lower() in ("true", "yes", "t", "1")
    rank_images = args.rank.lower() in ("true", "yes", "t", "1")

    return imgs, resize_images, rank_images
