import dataclasses
import os
import pprint
import time
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


def run_alignment(image_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Aligning image...")
    aligned_image = align_face(
        filepath=str(image_path), detector=detector, predictor=predictor
    )
    print(f"Finished aligning image: {image_path}")
    return aligned_image


def crop_image(image_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Cropping image...")
    cropped_image = crop_face(
        filepath=str(image_path), detector=detector, predictor=predictor
    )
    print(f"Finished cropping image: {image_path}")
    return cropped_image


def compute_transforms(aligned_path, cropped_path):
    download_dlib_models()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    print("Computing landmarks-based transforms...")
    res = get_stylegan_transform(
        str(cropped_path), str(aligned_path), detector, predictor
    )
    print("Done!")
    if res is None:
        print(f"Failed computing transforms on: {cropped_path}")
        return
    else:
        rotation_angle, translation, transform, inverse_transform = res
        return inverse_transform


model_path = "./pretrained_models/restyle_pSp_ffhq.pt"
image_path = Path("./images/face_image.jpg")
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

net, opts = load_encoder(checkpoint_path=model_path)
pprint.pprint(dataclasses.asdict(opts))

original_image = Image.open(image_path).convert("RGB")
original_image = original_image.resize((256, 256))

input_image = run_alignment(image_path)
cropped_image = crop_image(image_path)
joined = np.concatenate(
    [input_image.resize((256, 256)), cropped_image.resize((256, 256))], axis=1
)
Image.fromarray(joined)

images_dir = Path("./images")
images_dir.mkdir(exist_ok=True, parents=True)
cropped_path = images_dir / f"cropped_{image_path.name}"
aligned_path = images_dir / f"aligned_{image_path.name}"
cropped_image.save(cropped_path)
input_image.save(aligned_path)
landmarks_transform = compute_transforms(
    aligned_path=aligned_path, cropped_path=cropped_path
)

n_iters_per_batch = 3
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False
transformed_image = transform(input_image)

avg_image = get_average_image(net)

with torch.no_grad():
    tic = time.time()
    result_batch, result_latents = run_on_batch(
        inputs=transformed_image.unsqueeze(0).cuda().float(),
        net=net,
        opts=opts,
        avg_image=avg_image,
        landmarks_transform=torch.from_numpy(landmarks_transform).cuda().float(),
    )
    toc = time.time()
    print("Inference took {:.4f} seconds.".format(toc - tic))

editor = FaceEditor(
    stylegan_generator=net.decoder, generator_type=GeneratorType.ALIGNED
)

edit_direction = "age"
min_value = -5
max_value = 5

print(f"Performing edit for {edit_direction}...")
input_latent = torch.from_numpy(result_latents[0][-1]).unsqueeze(0).cuda()
edit_images, edit_latents = editor.edit(
    latents=input_latent,
    direction=edit_direction,
    factor_range=(min_value, max_value),
    user_transforms=landmarks_transform,
    apply_user_transformations=True,
)
print("Done!")


def prepare_edited_result(edit_images):
    if type(edit_images[0]) == list:
        edit_images = [image[0] for image in edit_images]
    res = np.array(edit_images[0].resize((512, 512)))
    for image in edit_images[1:]:
        res = np.concatenate([res, image.resize((512, 512))], axis=1)
    res = Image.fromarray(res).convert("RGB")
    return res


res = prepare_edited_result(edit_images)

res.save(f"edited_{image_path.name}")
