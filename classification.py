from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import yaml
from pathlib import Path
import gdown
import zipfile

tf.get_logger().setLevel('ERROR')


class Classification():
    def __init__(self, config_path: str = "config.yaml") -> None:
        # load config
        config = yaml.safe_load(open(config_path))
        self.class_name = config["class_names"]
        self.num_classes = len(self.class_name)

        # load model
        self.model_path = config["model_path"]
        self.model_id_gd = config["model_id_GD"]
        if not Path(self.model_path).exists():
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            print("Model does not exist")
            print("Downloading model...")
            gdown.download(id=self.model_id_gd, output=self.model_path, quiet=False)
            print("Model downloaded")
        self.model = tf.keras.models.load_model(self.model_path)

        # set image directory
        self.image_dir = config["image_dir"]
        if not Path(self.image_dir).exists():
            print("Image directory does not exist")
            self.image_dir_id_gd = config["image_dir_id_GD"]
            gdown.download(id=self.image_dir_id_gd, output=f'./data.zip', quiet=False)
            with zipfile.ZipFile('data.zip' , 'r') as zip_ref:
                zip_ref.extractall(Path('data'))
        self.image_size = config["image_size"]

        print("Model loaded")

        self.current_class = None
        self.visualize_layer_names = config["visualize_layer_names"]

    def prediction(self, image_path):
        img_array = self.load_and_transform_single_image(image_path)
        prediction = self.model.predict(img_array)
        score = tf.nn.softmax(prediction[0])

        self.current_class = self.class_name[tf.argmax(score)]

        return self.current_class, 100 * tf.reduce_max(score)

    def load_and_transform_single_image(self, image_path):
        """Loads and transforms a single image.

        Args:
            image_path: The path to the image file.

        Returns:
            The image as a 4D tensor.
        """
        img = tf.keras.utils.load_img(image_path, target_size=self.image_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        return img_array

    def retriev_image_from_class(self, image_path, return_number=3):
        if self.current_class is None:
            self.current_class = self.prediction(image_path)[0]

        class_dir = Path(self.image_dir) / self.current_class

        if not class_dir.exists():
            print("Class directory does not exist")
            return None

        image_paths = list(class_dir.glob("*.jpg"))
        if len(image_paths) < return_number:
            print("Not enough images in class")
            return None

        return list(map(str, np.random.choice(image_paths, return_number)))

    def visualize_feature_map(self, image_path, max_return_number=4):
        feature_maps = self._get_feature_map(image_path, max_return_number)
        fig_feature_maps = list(map(self._plot_feature_map, feature_maps))
        return fig_feature_maps

    def _get_feature_map(self, image_path, max_return_number=4):
        visualize_layer_names = self.visualize_layer_names[::-
                                                           1][:max_return_number]
        outputs = [self.model.get_layer(
            name).output for name in visualize_layer_names]
        model = tf.keras.Model(inputs=self.model.inputs, outputs=outputs)
        feature_maps = model.predict(
            self.load_and_transform_single_image(image_path))
        return feature_maps

    def _plot_feature_map(self, feature_maps):
        feature_maps = feature_maps[0, :, :, :]
        feature_maps_ = feature_maps.reshape(
            feature_maps.shape[0] * feature_maps.shape[1], feature_maps.shape[2])
        feature_maps_ = feature_maps_.sum(axis=0)
        feature_maps_indexs = feature_maps_.argsort()[-4:][::-1]
        feature_maps = feature_maps[:, :, feature_maps_indexs]
        fig = plt.figure(figsize=(10, 10))
        for i in range(feature_maps.shape[-1]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(feature_maps[:, :, i], cmap="gray")
            plt.axis("off")
        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        return fig


if __name__ == "__main__":
    classification = Classification()
    image_path = "flower_photo\\roses\\12240303_80d87f77a3_n.jpg"
    print(classification.visualize_feature_map(image_path))
