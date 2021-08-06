import numpy as np
from path import Path
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from .utils import score_utils


def main(imgpaths, resize_images=False, rank_images=False):
    target_size = (224, 224) if resize_images else None

    with tf.device("/CPU:0"):
        base_model = MobileNet(
            (None, None, 3), alpha=1, include_top=False, pooling="avg", weights=None
        )
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation="softmax")(x)

        model = Model(base_model.input, x)
        filepath = os.path.realpath(__file__)
        file_dirpath = os.path.dirname(filepath)
        model.load_weights(
            os.path.join(file_dirpath, "../../weights/mobilenet_weights.h5")
        )

        score_list = []

        for img_path in imgpaths:
            img = load_img(img_path, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = score_utils.mean_score(scores)
            std = score_utils.std_score(scores)

            file_name = Path(img_path).name.lower()
            score_list.append((file_name, mean))

            print("Evaluating : ", img_path)
            print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
            print()

        if rank_images:
            print("*" * 40, "Ranking Images", "*" * 40)
            score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

            for i, (name, score) in enumerate(score_list):
                print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))


if __name__ == "__main__":
    (imgs, resize_images, rank_images) = utils.cli.parse_args()
    main(imgs, resize_images, rank_images)
