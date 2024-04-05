import time
import numpy as np
import onnxruntime
import cv2
import onnx
from onnx import numpy_helper
from ..utils import face_align
import nvtx

import os

try:
    profile  # exists when line_profiler is running the script
except NameError:

    def profile(func):  # define a no-op "profile" function
        return func


class INSwapper:
    def __init__(self, model_file=None, session=None):
        self.model_file = model_file
        self.session = session
        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0
        # print('input mean and std:', model_file, self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        inputs = self.session.get_inputs()
        self.input_names = []
        for inp in inputs:
            self.input_names.append(inp.name)
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.output_names = output_names
        assert len(self.output_names) == 1
        output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        print("inswapper-shape:", self.input_shape)
        self.input_size = tuple(input_shape[2:4][::-1])
        self.filters = {}
        self.stream = cv2.cuda.Stream(1)

    def forward(self, img, latent):
        img = (img - self.input_mean) / self.input_std
        pred = self.session.run(
            self.output_names, {self.input_names[0]: img, self.input_names[1]: latent}
        )[0]
        return pred

    def __write_img(self, img, name):
        file_name = os.path.join("output", name)
        cv2.imwrite(file_name, img.astype(np.uint8))

    def __get_gaussain_filter(self, ksize):
        if ksize in self.filters:
            return self.filters[ksize]
        self.filters[ksize] = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC1, cv2.CV_8UC1, (ksize, ksize), ksize * 2
        )
        return self.filters[ksize]

    @nvtx.annotate("INSwapper.__get_CUDA")
    @profile
    def __get_CUDA(self, img, target_face, source_face, paste_back=True):
        nvtx.push_range("Data Preprocessing")
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        nvtx.pop_range()
        nvtx.push_range("Model Inference")
        pred = self.session.run(
            self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent}
        )[0]
        nvtx.pop_range()
        nvtx.push_range("Post Processing (GPU)")
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            nvtx.push_range("Upload Target Image")
            target_img_gpu = cv2.cuda.GpuMat()
            target_img_gpu.upload(img, stream=self.stream)
            nvtx.pop_range()

            nvtx.push_range("Image Mask")
            IM = cv2.invertAffineTransform(M)
            x, y = aimg.shape[:2]  # aimg.shape[0], aimg.shape[1]

            x_resized = int(x * 0.9)
            y_resized = int(y * 0.9)

            # To center `img_white`, you should calculate the top-left corner of the resized image
            center = (x // 2, y // 2)
            top_left_x = center[0] - x_resized // 2
            top_left_y = center[1] - y_resized // 2

            # Upload image mask as an alpha channel
            alpha_channel_gpu = cv2.cuda.GpuMat(y, x, cv2.CV_8UC1)

            # Set as black background
            alpha_channel_gpu.setTo(0, stream=self.stream)

            # Fill a white rectangle where the face will be placed
            alpha_channel_gpu.rowRange(top_left_y, top_left_y + y_resized).colRange(
                top_left_x, top_left_x + x_resized
            ).setTo(255, stream=self.stream)

            alpha_channel_gpu = cv2.cuda.warpAffine(
                alpha_channel_gpu, IM, target_img_gpu.size(), stream=self.stream
            )

            bgr_fake_gpu = cv2.cuda.GpuMat()
            bgr_fake_gpu.upload(bgr_fake, stream=self.stream)

            bgr_fake_gpu = cv2.cuda.warpAffine(
                bgr_fake_gpu, IM, target_img_gpu.size(), stream=self.stream
            )

            # Use math to calculate the new mask size
            # Original corner points
            points = np.array(
                [
                    [top_left_x, top_left_y, 1],
                    [top_left_x + x_resized, top_left_y, 1],
                    [top_left_x, top_left_y + y_resized, 1],
                    [top_left_x + x_resized, top_left_y + y_resized, 1],
                ]
            )

            # Transform points
            transformed_points = np.dot(
                points, IM.T
            )  # Note: IM.T for correct orientation

            # Calculate AABB of the transformed points
            min_x = np.min(transformed_points[:, 0])
            max_x = np.max(transformed_points[:, 0])
            min_y = np.min(transformed_points[:, 1])
            max_y = np.max(transformed_points[:, 1])

            # New dimensions
            new_width = max_x - min_x
            new_height = max_y - min_y

            # New mask size as sqrt(new_width * new_height)
            mask_size = int(np.sqrt(new_width * new_height))
            nvtx.pop_range()
            nvtx.push_range("Image Blurring")

            # Set kernel size and iterations for GPU filter options
            k = max(mask_size // 10, 10)
            if k % 2 == 0:  # Ensure kernel size is odd
                k += 1
            k = min(k, 31)  # Ensure kernel size is <= 61 on GPU
            iterations = (
                mask_size // 20 // k + 1
            )  # Apply iterations to make up for smaller kernel size vs. CPU

            # Blur edges of alpha channel, so source and target blend
            gpu_blur = self.__get_gaussain_filter(k)
            for _ in range(iterations):
                gpu_blur.apply(alpha_channel_gpu, alpha_channel_gpu, stream=self.stream)

            nvtx.pop_range()
            nvtx.push_range("Image Blending")
            # Create inverse alpha channel for blending background image
            scaler_gpu = cv2.cuda.GpuMat(alpha_channel_gpu.size(), cv2.CV_8UC1)
            scaler_gpu.setTo(255, stream=self.stream)
            inverse_alpha_channel_gpu = cv2.cuda.subtract(
                scaler_gpu, alpha_channel_gpu, stream=self.stream
            )

            # Set as Alpha channel
            blend_a = cv2.cuda.GpuMat(bgr_fake_gpu.size(), cv2.CV_8UC4)
            bgr_channels = cv2.cuda.split(bgr_fake_gpu, stream=self.stream)
            cv2.cuda.merge(
                [
                    bgr_channels[0],
                    bgr_channels[1],
                    bgr_channels[2],
                    alpha_channel_gpu,
                ],
                blend_a,
                stream=self.stream,
            )

            blend_b = cv2.cuda.GpuMat(target_img_gpu.size(), cv2.CV_8UC4)
            target_channels = cv2.cuda.split(target_img_gpu, stream=self.stream)
            cv2.cuda.merge(
                [
                    target_channels[0],
                    target_channels[1],
                    target_channels[2],
                    inverse_alpha_channel_gpu,
                ],
                blend_b,
                stream=self.stream,
            )

            fake_merged = cv2.cuda.alphaComp(
                blend_a, blend_b, cv2.cuda.ALPHA_PLUS, stream=self.stream
            )

            # Strip out the alpha channel - GPU Solution
            fake_merged_channels = cv2.cuda.split(fake_merged, stream=self.stream)
            cv2.cuda.merge(
                [
                    fake_merged_channels[0],
                    fake_merged_channels[1],
                    fake_merged_channels[2],
                ],
                fake_merged,
                stream=self.stream,
            )
            nvtx.pop_range()
            nvtx.pop_range()
            return fake_merged

    def get(self, img, target_face, source_face, paste_back=True):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return self.__get_CUDA(img, target_face, source_face, paste_back)
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)
        pred = self.session.run(
            self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent}
        )[0]
        img_fake = pred.transpose((0, 2, 3, 1))[0]
        bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
        if not paste_back:
            return bgr_fake, M
        else:
            target_img = img
            IM = cv2.invertAffineTransform(M)
            x, y = aimg.shape[:2]  # aimg.shape[0], aimg.shape[1]
            center = (x // 2, y // 2)
            x_resized = int(x * 0.9)
            y_resized = int(y * 0.9)

            # To center `img_white`, you should calculate the top-left corner of the resized image
            top_left_x = center[1] - x_resized // 2
            top_left_y = center[0] - y_resized // 2

            img_white = np.full((x_resized, y_resized), 255, dtype=np.float32)

            # Then create a new image where `img_white` is centered
            black_background_with_white_center = np.full((x, y), 0, dtype=np.float32)
            img_white_resized = np.full(
                (x_resized, y_resized), 255, dtype=np.float32
            )  # Your resized white image
            black_background_with_white_center[
                top_left_y : top_left_y + x_resized, top_left_x : top_left_x + y_resized
            ] = img_white_resized

            img_white = cv2.warpAffine(
                black_background_with_white_center,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            bgr_fake = cv2.warpAffine(
                bgr_fake,
                IM,
                (target_img.shape[1], target_img.shape[0]),
                borderValue=0.0,
            )
            img_white[img_white > 20] = 255

            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask == 255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h * mask_w))
            k = max(mask_size // 20, 5)
            kernel_size = (k, k)
            blur_size = tuple(2 * k + 1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            img_mask /= 255
            img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
            fake_merged = img_mask * bgr_fake + (1 - img_mask) * target_img.astype(
                np.float32
            )
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged

