import cv2
import albumentations as A


BBOX_PARAMS = dict(
    format="pascal_voc",
    min_visibility=0.2,
    label_fields=["labels"],
)


def pre_transform(image_size: int = 512):
    result = [
        A.LongestMaxSize(image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)),
    ]

    return A.Compose(result, bbox_params=BBOX_PARAMS)


def augmentations(image_size: int):
    channel_augs = [
        A.HueSaturationValue(p=0.5),
        A.ChannelShuffle(p=0.5),
    ]

    result = [
        # *pre_transform(image_size),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.7),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.Blur(blur_limit=3, p=0.7),
        ], p=0.5),
        A.OneOf(channel_augs),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
        ], p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.5,
            p=0.5
        ),
        A.RandomGamma(p=0.5),
        A.OneOf([
            A.MedianBlur(p=0.5),
            A.MotionBlur(p=0.5)
        ]),
        A.RandomGamma(gamma_limit=(85, 115), p=0.5),
    ]
    return A.Compose(result, bbox_params=BBOX_PARAMS)


def train_transform(image_size: int):
    result = A.Compose([
        *pre_transform(image_size),
        *augmentations(image_size),
    ], bbox_params=BBOX_PARAMS)
    return result


def valid_transform(image_size: int):
    result = A.Compose([
        *pre_transform(image_size),
    ], bbox_params=BBOX_PARAMS)
    return result


def infer_transform(image_size: int):
    result = A.Compose(pre_transform(image_size))
    return result
