import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType
from utils.alignment_utils import align_face, crop_face, get_stylegan_transform
from utils.inference_utils import get_average_image, load_encoder, run_on_batch

print("import dlib at last to fix crash bug")
import dlib


def download_dlib_models():
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print("Downloading files for aligning face image...")
        os.system(
            "wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        )
        os.system("bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2")
        print("Done.")


class FaceProcessor:
    def __init__(self):
        download_dlib_models()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()

    def align(self, dlib_image):
        print("Aligning image...")
        aligned_image = align_face(
            dlib_image=dlib_image, detector=self.detector, predictor=self.predictor
        )
        return aligned_image

    def crop(self, dlib_image):
        print("Cropping image...")
        cropped_image = crop_face(
            dlib_image=dlib_image, detector=self.detector, predictor=self.predictor
        )
        return cropped_image

    def compute_transforms(self, aligned_path, cropped_path):
        print("Computing landmarks-based transforms...")
        res = get_stylegan_transform(
            str(cropped_path), str(aligned_path), self.detector, self.predictor
        )
        print("Done!")
        if res is None:
            print(f"Failed computing transforms on: {cropped_path}")
            return
        else:
            rotation_angle, translation, transform, inverse_transform = res
            return inverse_transform


class FaceEditorWrapper:
    def __init__(
        self,
        n_iters_per_batch=3,
        resize_outputs=False,
        model_path="./pretrained_models/restyle_pSp_ffhq.pt",
    ) -> None:
        self.face_processor = FaceProcessor()
        self.net, self.opts = load_encoder(checkpoint_path=model_path)
        self.opts.n_iters_per_batch = n_iters_per_batch
        self.opts.resize_outputs = resize_outputs
        self.avg_image = get_average_image(self.net)
        self.editor = FaceEditor(
            stylegan_generator=self.net.decoder, generator_type=GeneratorType.ALIGNED
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __call__(self, original_image, edit_direction="age", min_value=-5, max_value=5):
        original_image = original_image.convert("RGB").resize((256, 256))
        image_path = Path("./images/face_image.jpg")
        original_image.save(image_path)
        dlib_image = dlib.load_rgb_image(image_path)
        input_image = self.face_processor.align(image_path)
        cropped_image = self.face_processor.crop(image_path)

        images_dir = Path("./images")
        images_dir.mkdir(exist_ok=True, parents=True)
        cropped_path = images_dir / f"cropped_{image_path.name}"
        aligned_path = images_dir / f"aligned_{image_path.name}"
        cropped_image.save(cropped_path)
        input_image.save(aligned_path)
        landmarks_transform = self.face_processor.compute_transforms(
            aligned_path=aligned_path, cropped_path=cropped_path
        )
        transformed_image = self.transform(input_image)

        with torch.no_grad():
            result_batch, result_latents = run_on_batch(
                inputs=transformed_image.unsqueeze(0).cuda().float(),
                net=self.net,
                opts=self.opts,
                avg_image=self.avg_image,
                landmarks_transform=torch.from_numpy(landmarks_transform)
                .cuda()
                .float(),
            )

        input_latent = torch.from_numpy(result_latents[0][-1]).unsqueeze(0).cuda()
        edit_images, edit_latents = self.editor.edit(
            latents=input_latent,
            direction=edit_direction,
            factor_range=(min_value, max_value),
            user_transforms=landmarks_transform,
            apply_user_transformations=True,
        )
        return edit_images


if __name__ == "__main__":
    image_path = Path("./images/face_image.jpg")
    original_image = Image.open(image_path)
    face_editor = FaceEditorWrapper()
    edit_images = face_editor(original_image)
