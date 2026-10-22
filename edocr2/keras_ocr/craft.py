"""The CRAFT text detector without TensorFlow: the PyTorch model (with the original weights
the Keras ones were converted from), the input scaling and the box finding. detection.py
imports these back, so `from edocr2.keras_ocr.detection import ...` still finds them."""

import typing

import cv2
import numpy as np

from . import tools


def compute_input(image):
    # should be RGB order
    image = image.astype("float32")
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image


def getBoxes(
    y_pred,
    detection_threshold=0.7,
    text_threshold=0.4,
    link_threshold=0.4,
    size_threshold=10,
):
    box_groups = []
    for y_pred_cur in y_pred:
        # Prepare data
        textmap = y_pred_cur[..., 0].copy()
        linkmap = y_pred_cur[..., 1].copy()
        img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(
            textmap, thresh=text_threshold, maxval=1, type=cv2.THRESH_BINARY
        )
        _, link_score = cv2.threshold(
            linkmap, thresh=link_threshold, maxval=1, type=cv2.THRESH_BINARY
        )
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.clip(text_score + link_score, 0, 1).astype("uint8"), connectivity=4
        )
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key]
                for key in [
                    cv2.CC_STAT_LEFT,
                    cv2.CC_STAT_TOP,
                    cv2.CC_STAT_WIDTH,
                    cv2.CC_STAT_HEIGHT,
                ]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)),
            )

            # Make rotated box from contour
            contours = cv2.findContours(
                segmap.astype("uint8"),
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )[-2]
            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)
        box_groups.append(np.array(boxes))
    return box_groups


# pylint: disable=import-error,too-few-public-methods
def build_torch_model(weights_path=None):
    from collections import namedtuple, OrderedDict

    import torch
    import torchvision

    def init_weights(modules):
        for m in modules:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    class vgg16_bn(torch.nn.Module):
        def __init__(self, pretrained=True, freeze=True):
            super().__init__()
            # We don't bother loading the pretrained VGG
            # because we're going to use the weights
            # at weights_path.
            vgg_pretrained_features = torchvision.models.vgg16_bn(weights=None).features
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            for x in range(12):  # conv2_2
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 19):  # conv3_3
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(19, 29):  # conv4_3
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(29, 39):  # conv5_3
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

            # fc6, fc7 without atrous conv
            self.slice5 = torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                torch.nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                torch.nn.Conv2d(1024, 1024, kernel_size=1),
            )

            if not pretrained:
                init_weights(self.slice1.modules())
                init_weights(self.slice2.modules())
                init_weights(self.slice3.modules())
                init_weights(self.slice4.modules())

            init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

            if freeze:
                for param in self.slice1.parameters():  # only first conv
                    param.requires_grad = False

        def forward(self, X):  # pylint: disable=arguments-differ
            h = self.slice1(X)
            h_relu2_2 = h
            h = self.slice2(h)
            h_relu3_2 = h
            h = self.slice3(h)
            h_relu4_3 = h
            h = self.slice4(h)
            h_relu5_3 = h
            h = self.slice5(h)
            h_fc7 = h
            vgg_outputs = namedtuple(
                "vgg_outputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"]
            )
            out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
            return out

    class double_conv(torch.nn.Module):
        def __init__(self, in_ch, mid_ch, out_ch):
            super().__init__()
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
                torch.nn.BatchNorm2d(mid_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
            )

        def forward(self, x):  # pylint: disable=arguments-differ
            x = self.conv(x)
            return x

    class CRAFT(torch.nn.Module):
        def __init__(self, pretrained=False, freeze=False):
            super().__init__()
            # Base network
            self.basenet = vgg16_bn(pretrained, freeze)
            # U network
            self.upconv1 = double_conv(1024, 512, 256)
            self.upconv2 = double_conv(512, 256, 128)
            self.upconv3 = double_conv(256, 128, 64)
            self.upconv4 = double_conv(128, 64, 32)

            num_class = 2
            self.conv_cls = torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(32, 16, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, 16, kernel_size=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, num_class, kernel_size=1),
            )

            init_weights(self.upconv1.modules())
            init_weights(self.upconv2.modules())
            init_weights(self.upconv3.modules())
            init_weights(self.upconv4.modules())
            init_weights(self.conv_cls.modules())

        def forward(self, x):  # pylint: disable=arguments-differ
            # Base network
            sources = self.basenet(x)
            # U network
            # pylint: disable=E1101
            y = torch.cat([sources[0], sources[1]], dim=1)

            y = self.upconv1(y)

            y = torch.nn.functional.interpolate(
                y, size=sources[2].size()[2:], mode="bilinear", align_corners=False
            )
            y = torch.cat([y, sources[2]], dim=1)
            y = self.upconv2(y)

            y = torch.nn.functional.interpolate(
                y, size=sources[3].size()[2:], mode="bilinear", align_corners=False
            )
            y = torch.cat([y, sources[3]], dim=1)
            y = self.upconv3(y)

            y = torch.nn.functional.interpolate(
                y, size=sources[4].size()[2:], mode="bilinear", align_corners=False
            )
            y = torch.cat([y, sources[4]], dim=1)
            # pylint: enable=E1101
            feature = self.upconv4(y)

            y = self.conv_cls(feature)

            return y.permute(0, 2, 3, 1), feature

    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    model = CRAFT(pretrained=True).eval()
    if weights_path is not None:
        model.load_state_dict(
            copyStateDict(torch.load(weights_path, map_location=torch.device("cpu")))
        )
    return model


PRETRAINED_WEIGHTS = {
    ("clovaai_general", True): {
        "url": "https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/craft_mlt_25k.pth",
        "filename": "craft_mlt_25k.pth",
        "sha256": "4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17",
    },
    ("clovaai_general", False): {
        "url": "https://github.com/faustomorales/keras-ocr/releases/download/v0.8.4/craft_mlt_25k.h5",
        "filename": "craft_mlt_25k.h5",
        "sha256": "7283ce2ff05a0617e9740c316175ff3bacdd7215dbdf1a726890d5099431f899",
    },
}


class TorchDetector:
    """The same CRAFT detector in PyTorch, with the original PyTorch weights that the Keras
    weights were converted from. Its output matches Detector's to rounding, and `detect` takes
    the same arguments, so the two are interchangeable in the pipeline.

    On this repository's test page the Keras model took about two minutes per detection patch
    on a CPU; this takes under a second on a CPU and under 0.1 s on a GPU.

    Args:
        weights: Only `clovaai_general` is available.
        device: "cuda", "cpu", or None for the GPU when there is one.
    """

    def __init__(self, weights="clovaai_general", device=None):
        import torch

        weights_config = PRETRAINED_WEIGHTS[(weights, True)]
        weights_path = tools.download_and_verify(
            url=weights_config["url"],
            filename=weights_config["filename"],
            sha256=weights_config["sha256"],
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_torch_model(weights_path=weights_path).to(self.device).eval()

    def detect(
        self,
        images: typing.List[typing.Union[np.ndarray, str]],
        detection_threshold=0.7,
        text_threshold=0.4,
        link_threshold=0.4,
        size_threshold=10,
        **kwargs,
    ):
        """Detect text in a list of images of the same size; see Detector.detect."""
        import torch

        X = np.array([compute_input(tools.read(image)) for image in images], dtype="float32")
        with torch.no_grad():
            y = self.model(torch.from_numpy(X).permute(0, 3, 1, 2).to(self.device))[0]
        return getBoxes(
            y.cpu().numpy(),
            detection_threshold=detection_threshold,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            size_threshold=size_threshold,
        )
