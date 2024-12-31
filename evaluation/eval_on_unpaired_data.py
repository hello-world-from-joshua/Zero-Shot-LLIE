# =================================================================
# Evalution code for unpaired (DICM, LIME, MEF, NPE, VV) datasets.
# To use, please replace directory_mapping with your paths.
# Command: python eval_on_unpaired_data.py
# --resize if resizing to the original dimension.
# =================================================================

import os
import pyiqa
import argparse
from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument("--resize", action="store_true", help="Resize output images to match original input dimension if True.")
args = parser.parse_args()

directory_mapping = {
    "path to our DICM output": "path to DICM data",
    "path to our LIME output": "path to LIME data",
    "path to our MEF output": "path to MEF data",
    "path to our NPE output": "path to NPE data",
    "path to our VV output": "path to VV data",

    "path to <baseline method1> DICM output": "path to DICM data",
    "path to <baseline method1> LIME output": "path to LIME data",
    "path to <baseline method1> MEF output": "path to MEF data",
    "path to <baseline method1> NPE output": "path to NPE data",
    "path to <baselnie method1> VV output": "path to VV data",

    "add more below as above"
}

niqe = pyiqa.create_metric("niqe_matlab")
piqe = pyiqa.create_metric("piqe")
ilniqe = pyiqa.create_metric("ilniqe")

print(f"\nNIQE (lower better): {niqe.lower_better} (range): {niqe.score_range}")
print(f"PIQE (lower better): {piqe.lower_better} (range): {piqe.score_range}")
print(f"ILNIQE (lower better): {ilniqe.lower_better} (range): {ilniqe.score_range}")

transform = transforms.Compose([
    transforms.ToTensor(),
])

for dir_path, test_data_path in directory_mapping.items():
    print(f"\nProcessing directory: {dir_path}")

    results = []

    for filename in os.listdir(dir_path):
        main_img_path = os.path.join(dir_path, filename)
        test_img_path = os.path.join(test_data_path, filename)

        with Image.open(main_img_path) as main_img, Image.open(test_img_path) as test_img:
            test_width, test_height = test_img.size
            if args.resize:
                main_img_resized = main_img.resize((test_width, test_height), Image.LANCZOS)
            else:
                main_img_resized = main_img

            img_tensor = transform(main_img_resized).unsqueeze(0) 

            niqe_score = niqe(img_tensor)
            piqe_score = piqe(img_tensor)
            ilniqe_score = ilniqe(img_tensor)

            results.append((filename, niqe_score.item(), piqe_score.item(), ilniqe_score.item()))

    average_niqe_score = sum(score[1] for score in results) / len(results)
    average_piqe_score = sum(score[2] for score in results) / len(results)
    average_ilniqe_score = sum(score[3] for score in results) / len(results)
            
    print(f"Avg NIQE: {round(average_niqe_score, 3)}")
    print(f"Avg PIQE: {round(average_piqe_score, 3)}")
    print(f"Avg ILNIQE: {round(average_ilniqe_score, 3)}")