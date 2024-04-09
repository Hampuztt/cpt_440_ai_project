import os
import requests
import zipfile
from io import BytesIO


# URLs for the RecipeQA dataset
def get_datasets():
    datasets = {
        "training_set": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/train.json",
        "validation_set": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/val.json",
        "test_set": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/test.json",
        "images": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/images.zip",
        "validation_set_recipies": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/recipes.zip",
    }
    return datasets


def get_file_extension(url):
    if url.endswith(".zip"):
        return ".zip"
    elif url.endswith(".json"):
        return ".json"
    else:
        return ""  # default if not found


def main():
    datasets = get_datasets()

    project_dir = os.getcwd()
    dataset_dir = os.path.join(project_dir, "datasets")

    # Ensure the dataset directory exists
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Checking for missing files and their sizes
    total_size_to_download = 0
    files_to_download = {}

    for name, url in datasets.items():
        ext = get_file_extension(url)
        file_path = os.path.join(dataset_dir, f"{name}{ext}")
        if not os.path.exists(file_path):
            with requests.head(url, allow_redirects=True) as response:
                if response.status_code == 200:
                    size = int(response.headers.get("content-length", 0))
                    total_size_to_download += size
                    files_to_download[name] = (url, size)
                    print(f"{name}{ext}: {size / (1024 * 1024):.2f} MB (missing)")
                else:
                    print(
                        f"Error accessing {name}{ext}: Status code {response.status_code}"
                    )
        else:
            print(f"{name}{ext}: already downloaded.")

    # Prompting user for downloading the missing files
    if files_to_download:
        print(f"Total size to download: {total_size_to_download / (1024 ** 2):.2f} MB")
        confirmation = (
            input("Do you want to download all missing files? (yes/no): ")
            .strip()
            .lower()
        )
        if confirmation == "yes":
            for name, (url, size) in files_to_download.items():
                ext = get_file_extension(url)
                file_path = os.path.join(dataset_dir, f"{name}{ext}")
                print(f"Downloading {name}{ext}...")
                response = requests.get(url, stream=True)
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {name}{ext} to {dataset_dir}")
                if ext == ".zip":
                    print(f"Extracting {name}{ext}...")
                    with zipfile.ZipFile(file_path, "r") as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    print(f"Extracted {name} in {dataset_dir}")
    else:
        print("All files are already downloaded.")
        exit()

    print("Setup process is complete.")


if __name__ == "__main__":
    main()
