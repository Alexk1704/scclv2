# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow import keras

from keras_cv import layers

from .ContrastiveTrainer import ContrastiveTrainer



class SimCLRTrainer(ContrastiveTrainer):
    """Creates a SimCLRTrainer.

    References:
        - [SimCLR paper](https://arxiv.org/pdf/2002.05709)

    Args:
        encoder: a `keras.Model` to be pre-trained. In most cases, this encoder
            should not include a top dense layer.
        augmenter: a SimCLRAugmenter layer to randomly augment input
            images for contrastive learning
        projection_width: the width of the two-layer dense model used for
            projection in the SimCLR paper
    """

    def __init__(self, encoder, augmenter, projection_width=128, **kwargs):
        super().__init__(
            encoder=encoder,
            augmenter=augmenter,
            projector=keras.Sequential(
                [
                    keras.layers.Dense(projection_width, activation="relu"),
                    keras.layers.Dense(projection_width),
                    keras.layers.BatchNormalization(),
                ],
                name="projector",
            ),
            **kwargs,
        )



class SimCLRAugmenter(keras.Sequential):
    def __init__(
        self,
        value_range=(0, 1),
        height=128,
        width=128,
        crop_area_factor=(0.08, 1.0),
        aspect_ratio_factor=(3 / 4, 4 / 3),
        grayscale_rate=0.2,
        color_jitter_rate=0.8,
        brightness_factor=0.2,
        contrast_factor=0.8,
        saturation_factor=(0.3, 0.7),
        hue_factor=0.2,
        **kwargs,
    ):
        return super().__init__(
            [
                # keras.layers.Input(shape=input_shape),
                layers.Rescaling(scale=1.0 / 255), # [0,1] scaling
                layers.RandomFlip("horizontal"),
                layers.RandomCropAndResize(
                    target_size=(height, width), # (input_shape[0], input_shape[1])
                    crop_area_factor=crop_area_factor,
                    aspect_ratio_factor=aspect_ratio_factor,
                ),
                layers.RandomApply(
                    layers.Grayscale(output_channels=3),
                    rate=grayscale_rate,
                ),
                layers.RandomApply(
                    layers.RandomColorJitter(
                        value_range=value_range,
                        brightness_factor=brightness_factor,
                        contrast_factor=contrast_factor,
                        saturation_factor=saturation_factor,
                        hue_factor=hue_factor,
                    ),
                    rate=color_jitter_rate,
                ),
            ],
            **kwargs,
        )
