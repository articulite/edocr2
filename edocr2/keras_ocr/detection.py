# pylint: disable=invalid-name,too-many-locals,line-too-long,no-else-raise,too-many-arguments,no-self-use,too-many-statements,stop-iteration-return,import-outside-toplevel
import typing

# The PyTorch portions of this code are subject to the following copyright notice.
# Copyright (c) 2019-present NAVER Corp.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cv2
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efficientnet
from tensorflow import keras

from . import tools
from .craft import PRETRAINED_WEIGHTS, TorchDetector, build_torch_model, compute_input, getBoxes  # noqa: F401




def invert_input(X):
    X = X.copy()
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    X *= variance * 255
    X += mean * 255
    return X.clip(0, 255).astype("uint8")


def get_gaussian_heatmap(size=512, distanceRatio=3.34):
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    x, y = np.meshgrid(v, v)
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype("uint8")


def upconv(x, n, filters):
    x = keras.layers.Conv2D(
        filters=filters, kernel_size=1, strides=1, name=f"upconv{n}.conv.0"
    )(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.9, name=f"upconv{n}.conv.1"
    )(x)
    x = keras.layers.Activation("relu", name=f"upconv{n}.conv.2")(x)
    x = keras.layers.Conv2D(
        filters=filters // 2,
        kernel_size=3,
        strides=1,
        padding="same",
        name=f"upconv{n}.conv.3",
    )(x)
    x = keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.9, name=f"upconv{n}.conv.4"
    )(x)
    x = keras.layers.Activation("relu", name=f"upconv{n}.conv.5")(x)
    return x


def make_vgg_block(x, filters, n, prefix, pooling=True):
    x = keras.layers.Conv2D(
        filters=filters,
        strides=(1, 1),
        kernel_size=(3, 3),
        padding="same",
        name=f"{prefix}.{n}",
    )(x)
    x = keras.layers.BatchNormalization(
        momentum=0.1, epsilon=1e-5, axis=-1, name=f"{prefix}.{n+1}"
    )(x)
    x = keras.layers.Activation("relu", name=f"{prefix}.{n+2}")(x)
    if pooling:
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2), padding="valid", strides=(2, 2), name=f"{prefix}.{n+3}"
        )(x)
    return x


def compute_maps(heatmap, image_height, image_width, lines):
    assert image_height % 2 == 0, "Height must be an even number"
    assert image_width % 2 == 0, "Width must be an even number"

    textmap = np.zeros((image_height // 2, image_width // 2)).astype("float32")
    linkmap = np.zeros((image_height // 2, image_width // 2)).astype("float32")

    src = np.array(
        [
            [0, 0],
            [heatmap.shape[1], 0],
            [heatmap.shape[1], heatmap.shape[0]],
            [0, heatmap.shape[0]],
        ]
    ).astype("float32")

    for line in lines:
        line, orientation = tools.fix_line(line)
        previous_link_points = None
        for [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], c in line:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(
                lambda v: max(v, 0), [x1, y1, x2, y2, x3, y3, x4, y4]
            )
            if c == " ":
                previous_link_points = None
                continue
            yc = (y4 + y1 + y3 + y2) / 4
            xc = (x1 + x2 + x3 + x4) / 4
            if orientation == "horizontal":
                current_link_points = (
                    np.array(
                        [
                            [(xc + (x1 + x2) / 2) / 2, (yc + (y1 + y2) / 2) / 2],
                            [(xc + (x3 + x4) / 2) / 2, (yc + (y3 + y4) / 2) / 2],
                        ]
                    )
                    / 2
                )
            else:
                current_link_points = (
                    np.array(
                        [
                            [(xc + (x1 + x4) / 2) / 2, (yc + (y1 + y4) / 2) / 2],
                            [(xc + (x2 + x3) / 2) / 2, (yc + (y2 + y3) / 2) / 2],
                        ]
                    )
                    / 2
                )
            character_points = (
                np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype("float32") / 2
            )
            # pylint: disable=unsubscriptable-object
            if previous_link_points is not None:
                if orientation == "horizontal":
                    link_points = np.array(
                        [
                            previous_link_points[0],
                            current_link_points[0],
                            current_link_points[1],
                            previous_link_points[1],
                        ]
                    )
                else:
                    link_points = np.array(
                        [
                            previous_link_points[0],
                            previous_link_points[1],
                            current_link_points[1],
                            current_link_points[0],
                        ]
                    )
                ML = cv2.getPerspectiveTransform(
                    src=src,
                    dst=link_points.astype("float32"),
                )
                linkmap += cv2.warpPerspective(
                    heatmap, ML, dsize=(linkmap.shape[1], linkmap.shape[0])
                ).astype("float32")
            MA = cv2.getPerspectiveTransform(
                src=src,
                dst=character_points,
            )
            textmap += cv2.warpPerspective(
                heatmap, MA, dsize=(textmap.shape[1], textmap.shape[0])
            ).astype("float32")
            # pylint: enable=unsubscriptable-object
            previous_link_points = current_link_points
    return (
        np.concatenate(
            [textmap[..., np.newaxis], linkmap[..., np.newaxis]], axis=2
        ).clip(0, 255)
        / 255
    )


def map_to_rgb(y):
    return (
        np.concatenate([y, np.zeros((y.shape[0], y.shape[1], 1))], axis=-1) * 255
    ).astype("uint8")




class UpsampleLike(keras.layers.Layer):
    """Keras layer for upsampling a Tensor to be the same shape as another Tensor."""

    # pylint:disable=unused-argument
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == "channels_first":
            raise NotImplementedError
        else:
            # pylint: disable=no-member
            return tf.compat.v1.image.resize_bilinear(
                source, size=(target_shape[1], target_shape[2]), half_pixel_centers=True
            )

    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == "channels_first":
            raise NotImplementedError
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


def build_vgg_backbone(inputs):
    x = make_vgg_block(inputs, filters=64, n=0, pooling=False, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=64, n=3, pooling=True, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=128, n=7, pooling=False, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=128, n=10, pooling=True, prefix="basenet.slice1")
    x = make_vgg_block(x, filters=256, n=14, pooling=False, prefix="basenet.slice2")
    x = make_vgg_block(x, filters=256, n=17, pooling=False, prefix="basenet.slice2")
    x = make_vgg_block(x, filters=256, n=20, pooling=True, prefix="basenet.slice3")
    x = make_vgg_block(x, filters=512, n=24, pooling=False, prefix="basenet.slice3")
    x = make_vgg_block(x, filters=512, n=27, pooling=False, prefix="basenet.slice3")
    x = make_vgg_block(x, filters=512, n=30, pooling=True, prefix="basenet.slice4")
    x = make_vgg_block(x, filters=512, n=34, pooling=False, prefix="basenet.slice4")
    x = make_vgg_block(x, filters=512, n=37, pooling=False, prefix="basenet.slice4")
    x = make_vgg_block(x, filters=512, n=40, pooling=True, prefix="basenet.slice4")
    vgg = keras.models.Model(inputs=inputs, outputs=x)
    return [
        vgg.get_layer(slice_name).output
        for slice_name in [
            "basenet.slice1.12",
            "basenet.slice2.19",
            "basenet.slice3.29",
            "basenet.slice4.38",
        ]
    ]


def build_efficientnet_backbone(inputs, backbone_name, imagenet):
    backbone = getattr(efficientnet, backbone_name)(
        include_top=False, input_tensor=inputs, weights="imagenet" if imagenet else None
    )
    return [
        backbone.get_layer(slice_name).output
        for slice_name in [
            "block2a_expand_activation",
            "block3a_expand_activation",
            "block4a_expand_activation",
            "block5a_expand_activation",
        ]
    ]


def build_keras_model(weights_path: str = None, backbone_name="vgg"):
    inputs = keras.layers.Input((None, None, 3))

    if backbone_name == "vgg":
        s1, s2, s3, s4 = build_vgg_backbone(inputs)
    elif "efficientnet" in backbone_name.lower():
        s1, s2, s3, s4 = build_efficientnet_backbone(
            inputs=inputs, backbone_name=backbone_name, imagenet=weights_path is None
        )
    else:
        raise NotImplementedError

    s5 = keras.layers.MaxPooling2D(
        pool_size=3, strides=1, padding="same", name="basenet.slice5.0"
    )(s4)
    s5 = keras.layers.Conv2D(
        1024,
        kernel_size=(3, 3),
        padding="same",
        strides=1,
        dilation_rate=6,
        name="basenet.slice5.1",
    )(s5)
    s5 = keras.layers.Conv2D(
        1024, kernel_size=1, strides=1, padding="same", name="basenet.slice5.2"
    )(s5)

    y = keras.layers.Concatenate()([s5, s4])
    y = upconv(y, n=1, filters=512)
    y = UpsampleLike()([y, s3])
    y = keras.layers.Concatenate()([y, s3])
    y = upconv(y, n=2, filters=256)
    y = UpsampleLike()([y, s2])
    y = keras.layers.Concatenate()([y, s2])
    y = upconv(y, n=3, filters=128)
    y = UpsampleLike()([y, s1])
    y = keras.layers.Concatenate()([y, s1])
    features = upconv(y, n=4, filters=64)

    y = keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="same", name="conv_cls.0"
    )(features)
    y = keras.layers.Activation("relu", name="conv_cls.1")(y)
    y = keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="same", name="conv_cls.2"
    )(y)
    y = keras.layers.Activation("relu", name="conv_cls.3")(y)
    y = keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same", name="conv_cls.4"
    )(y)
    y = keras.layers.Activation("relu", name="conv_cls.5")(y)
    y = keras.layers.Conv2D(
        filters=16, kernel_size=1, strides=1, padding="same", name="conv_cls.6"
    )(y)
    y = keras.layers.Activation("relu", name="conv_cls.7")(y)
    y = keras.layers.Conv2D(
        filters=2, kernel_size=1, strides=1, padding="same", name="conv_cls.8"
    )(y)
    if backbone_name != "vgg":
        y = keras.layers.Activation("sigmoid")(y)
    model = keras.models.Model(inputs=inputs, outputs=y)
    if weights_path is not None:
        if weights_path.endswith(".h5"):
            model.load_weights(weights_path)
        elif weights_path.endswith(".pth"):
            assert (
                backbone_name == "vgg"
            ), "PyTorch weights only allowed with VGG backbone."
            load_torch_weights(model=model, weights_path=weights_path)
        else:
            raise NotImplementedError(f"Cannot load weights from {weights_path}")
    return model


# pylint: disable=import-error
def load_torch_weights(model, weights_path):
    import torch

    pretrained = torch.load(weights_path, map_location=torch.device("cpu"))
    layer_names = list(
        set(
            ".".join(k.split(".")[1:-1])
            for k in pretrained.keys()
            if k.split(".")[-1] != "num_batches_tracked"
        )
    )
    for layer_name in layer_names:
        try:
            layer = model.get_layer(layer_name)
        except Exception:  # pylint: disable=broad-except
            print("Skipping", layer.name)
            continue
        if isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, running_mean, running_std = [
                pretrained[k].numpy()
                for k in [
                    f"module.{layer_name}.weight",
                    f"module.{layer_name}.bias",
                    f"module.{layer_name}.running_mean",
                    f"module.{layer_name}.running_var",
                ]
            ]
            layer.set_weights([gamma, beta, running_mean, running_std])
        elif isinstance(layer, keras.layers.Conv2D):
            weights, bias = [
                pretrained[k].numpy()
                for k in [f"module.{layer_name}.weight", f"module.{layer_name}.bias"]
            ]
            layer.set_weights([weights.transpose(2, 3, 1, 0), bias])

        else:
            raise NotImplementedError

    for layer in model.layers:
        if isinstance(layer, (keras.layers.BatchNormalization, keras.layers.Conv2D)):
            assert layer.name in layer_names






class Detector:
    """A text detector using the CRAFT architecture.

    Args:
        weights: The weights to use for the model. Currently, only `clovaai_general`
            is supported.
        load_from_torch: Whether to load the weights from the original PyTorch weights.
        optimizer: The optimizer to use for training the model.
        backbone_name: The backbone to use. Currently, only 'vgg' is supported.
    """

    def __init__(
        self,
        weights="clovaai_general",
        load_from_torch=False,
        optimizer="adam",
        backbone_name="vgg",
    ):
        if weights is not None:
            pretrained_key = (weights, load_from_torch)
            assert backbone_name == "vgg", "Pretrained weights available only for VGG."
            assert (
                pretrained_key in PRETRAINED_WEIGHTS
            ), "Selected weights configuration not found."
            weights_config = PRETRAINED_WEIGHTS[pretrained_key]
            weights_path = tools.download_and_verify(
                url=weights_config["url"],
                filename=weights_config["filename"],
                sha256=weights_config["sha256"],
            )
        else:
            weights_path = None
        self.model = build_keras_model(
            weights_path=weights_path, backbone_name=backbone_name
        )
        self.model.compile(loss="mse", optimizer=optimizer)

    def get_batch_generator(
        self,
        image_generator,
        batch_size=8,
        heatmap_size=512,
        heatmap_distance_ratio=1.5,
    ):
        """Get a generator of X, y batches to train the detector.

        Args:
            image_generator: A generator with the same signature as
                keras_ocr.tools.get_image_generator. Optionally, a third
                entry in the tuple (beyond image and lines) can be provided
                which will be interpreted as the sample weight.
            batch_size: The size of batches to generate.
            heatmap_size: The size of the heatmap to pass to get_gaussian_heatmap
            heatmap_distance_ratio: The distance ratio to pass to
                get_gaussian_heatmap. The larger the value, the more tightly
                concentrated the heatmap becomes.
        """
        heatmap = get_gaussian_heatmap(
            size=heatmap_size, distanceRatio=heatmap_distance_ratio
        )
        while True:
            batch = [next(image_generator) for n in range(batch_size)]
            images = np.array([entry[0] for entry in batch])
            line_groups = [entry[1] for entry in batch]
            X = compute_input(images)
            # pylint: disable=unsubscriptable-object
            y = np.array(
                [
                    compute_maps(
                        heatmap=heatmap,
                        image_height=images.shape[1],
                        image_width=images.shape[2],
                        lines=lines,
                    )
                    for lines in line_groups
                ]
            )
            # pylint: enable=unsubscriptable-object
            if len(batch[0]) == 3:
                sample_weights = np.array([sample[2] for sample in batch])
                yield X, y, sample_weights
            else:
                yield X, y

    def detect(
        self,
        images: typing.List[typing.Union[np.ndarray, str]],
        detection_threshold=0.7,
        text_threshold=0.4,
        link_threshold=0.4,
        size_threshold=10,
        **kwargs,
    ):
        """Recognize the text in a set of images.

        Args:
            images: Can be a list of numpy arrays of shape HxWx3 or a list of
                filepaths.
            link_threshold: This is the same as `text_threshold`, but is applied to the
                link map instead of the text map.
            detection_threshold: We want to avoid including boxes that may have
                represented large regions of low confidence text predictions. To do this,
                we do a final check for each word box to make sure the maximum confidence
                value exceeds some detection threshold. This is the threshold used for
                this check.
            text_threshold: When the text map is processed, it is converted from confidence
                (float from zero to one) values to classification (0 for not text, 1 for
                text) using binary thresholding. The threshold value determines the
                breakpoint at which a value is converted to a 1 or a 0. For example, if
                the threshold is 0.4 and a value for particular point on the text map is
                0.5, that value gets converted to a 1. The higher this value is, the less
                likely it is that characters will be merged together into a single word.
                The lower this value is, the more likely it is that non-text will be detected.
                Therein lies the balance.
            size_threshold: The minimum area for a word.
        """
        images = [compute_input(tools.read(image)) for image in images]
        boxes = getBoxes(
            self.model.predict(np.array(images), **kwargs),
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            size_threshold=size_threshold,
        )
        return boxes


