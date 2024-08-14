import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
import numpy as np


class ResizeCrop(torch.nn.Module):
    def __init__(self,
                 p: float,
                 input_size: int,
                 weights: Sequence[float] = [
                     1, 9.04788032, 8.68207691, 12.9632271]) -> None:
        """ResizeCrop class.

        Args:
            p: Probability of applying a random size of crop 
                (different from the input size).
            input_size: Size of the input image after cropping and resizing.
            weights: Weights to be applied to the different classes in the
                cropped masks. We have one weight per damage class (4). 
                The default values are the inverse of the frequency of each
                class in the training/validation set with no contamination
                (see generalization section in the paper)
        """

        super().__init__()
        self.p = p
        self.input_size = input_size
        self.w = weights

    def forward(self,
                tensor: torch.Tensor) -> torch.Tensor:

        crop_size = self.input_size
        if random.random() > self.p:
            crop_size = random.randint(
                int(self.input_size / 1.15), int(self.input_size / 0.85))

            # We need to check that the tensor has the expected shape (17, H, W)
            # 17 = img (6 channels) + mask (5 channels) + aug_img (6 channels) (labeled data)
            assert tensor.shape[0] == 17

            # Separate img, msk, and aug_img from the concatenated tensor
            img = tensor[:6, ...]      # First 6 channels are img
            msk = tensor[6:11, ...]    # Next 5 channels are msk
            aug_img = tensor[11:, ...] # Remaining 6 channels are aug_img

            bst_x0 = random.randint(0, tensor.shape[1] - crop_size)
            bst_y0 = random.randint(0, tensor.shape[2] - crop_size)
            bst_sc = -1
            try_cnt = random.randint(1, 10)
            for _ in range(try_cnt):
                x0 = random.randint(0, tensor.shape[1] - crop_size)
                y0 = random.randint(0, tensor.shape[2] - crop_size)
                # We try to get more of certain classes in the cropped masks.
                _sc = msk[2, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[1] + \
                    msk[3, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[2] + \
                    msk[4, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[3] + \
                    msk[1, y0:y0 + crop_size, x0:x0 + crop_size].sum() * self.w[0]
                if _sc > bst_sc:
                    bst_sc = _sc
                    bst_x0 = x0
                    bst_y0 = y0
            x0 = bst_x0
            y0 = bst_y0

            # Apply the crop to the entire tensor (img, msk, aug_img)
            tensor = tensor[:, y0:y0 + crop_size, x0:x0 + crop_size]

        # Resize the cropped tensor back to the input size
        tensor = TF.resize(img=tensor,
                        size=[self.input_size, self.input_size],
                        interpolation=transforms.InterpolationMode.NEAREST,
                        antialias=True)
        #print(f'shape of resize crop output={tensor.shape}')
        return tensor
    

class FusionAugmentation:
    def __init__(self):
        pass

    def contrast_enhancement(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, strength=1.5):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(strength + 1) * image - float(strength) * blurred
        sharpened = np.maximum(sharpened, 0)
        sharpened = np.minimum(sharpened, 255)
        sharpened = sharpened.round().astype(np.uint8)
        return sharpened

    def sobel_edge_detection(self, image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # x direction
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # y direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude = np.uint8(magnitude)
        return magnitude

    def image_augmentation_rgb(self, image):
        channels = cv2.split(image)
        augmented_channels = []
        for channel in channels:
            enhanced_channel = self.contrast_enhancement(channel)
            unsharp_image = self.unsharp_mask(enhanced_channel)
            edge_image = self.sobel_edge_detection(unsharp_image)
            augmented_channels.append(edge_image)
        augmented_image = cv2.merge(augmented_channels)
        return augmented_image

    def __call__(self, image):
        augmented_image = self.image_augmentation_rgb(image)
        tensor_image = torch.tensor(augmented_image, dtype=torch.float32)
        tensor_image = tensor_image / 255.0
        tensor_image = tensor_image.permute(2, 0, 1)
        #print(f'tensor shape of image after augmentation= {tensor_image.shape}')
        return tensor_image
    
# class FusionAugmentation(torch.nn.Module):
#     def __init__(self, p: float, input_size: int, weights: Sequence[float] = [
#                      1, 9.04788032, 8.68207691, 12.9632271]):
#         super().__init__()
#         self.resize_crop = ResizeCrop(p, input_size, weights)

#     def contrast_enhancement(self, tensor):
#         # Convert tensor to NumPy array
#         image = tensor.permute(1, 2, 0).cpu().numpy()  # Change from (C, H, W) to (H, W, C)
#         channels = cv2.split(image)
#         enhanced_channels = []
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         for channel in channels:
#             enhanced_channel = clahe.apply(channel)
#             enhanced_channels.append(enhanced_channel)
#         enhanced_image = cv2.merge(enhanced_channels)
#         # Convert back to tensor
#         enhanced_tensor = torch.tensor(enhanced_image, dtype=torch.float32).permute(2, 0, 1)
#         return enhanced_tensor

#     def unsharp_mask(self, tensor, kernel_size=(5, 5), sigma=1.0, strength=1.5):
#         # Convert tensor to NumPy array
#         image = tensor.permute(1, 2, 0).cpu().numpy()
#         blurred = cv2.GaussianBlur(image, kernel_size, sigma)
#         sharpened = image * (strength + 1) - blurred * strength
#         sharpened = np.clip(sharpened, 0, 255)
#         sharpened = sharpened.round().astype(np.uint8)
#         # Convert back to tensor
#         sharpened_tensor = torch.tensor(sharpened, dtype=torch.float32).permute(2, 0, 1)
#         return sharpened_tensor

#     def sobel_edge_detection(self, tensor):
#         # Convert tensor to NumPy array
#         image = tensor.permute(1, 2, 0).cpu().numpy()
#         sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
#         sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
#         magnitude = np.sqrt(sobelx**2 + sobely**2)
#         magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
#         magnitude = magnitude.astype(np.uint8)
#         # Convert back to tensor
#         magnitude_tensor = torch.tensor(magnitude, dtype=torch.float32).unsqueeze(0)
#         return magnitude_tensor

#     def image_augmentation_rgb(self, tensor):
#         channels = tensor.split(1, dim=0)
#         augmented_channels = []
#         for channel in channels:
#             enhanced_channel = self.contrast_enhancement(channel.squeeze(0))
#             unsharp_image = self.unsharp_mask(enhanced_channel)
#             edge_image = self.sobel_edge_detection(unsharp_image)
#             augmented_channels.append(edge_image)
#         augmented_image = torch.cat(augmented_channels, dim=0)
#         return augmented_image

#     def forward(self, tensor):
#         print(f'shape of input tensor to augmentation {tensor.shape}')
#         augmented_tensor = self.image_augmentation_rgb(tensor)
#         tensor_image = augmented_tensor / 255.0  # Normalize to [0, 1]
#         print(f'tensor shape of image after augmentation= {tensor_image.shape}')
#         resized_tensor_image = self.resize_crop(tensor_image)
#         print(f'tensor shape  after resizing {resized_tensor_image}')
#         return resized_tensor_image