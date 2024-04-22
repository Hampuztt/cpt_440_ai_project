import os
import requests
import zipfile
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from typing import List
import json

from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    Document,
)


# URLs for the RecipeQA dataset
def get_datasets():
    datasets = {
        # "training_set": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/train.json",
        # "validation_set": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/val.json",
        "test_set": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/test.json",
        # "images": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/images.zip",
        # "validation_set_recipies": "https://vision.cs.hacettepe.edu.tr/files/recipeqa/recipes.zip",
    }
    return datasets


def get_file_extension(url):
    if url.endswith(".zip"):
        return ".zip"
    elif url.endswith(".json"):
        return ".json"
    else:
        return ""  # default if not found


def download_dataset():
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
        if confirmation == "yes" or "y":
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
        print("Setup cancelled")
        exit()

    print("All files downloaded.")


def text_ordinal(n):
    if n % 100 in (11, 12, 13):  # Special case for 11th to 13th
        return f"{n}th"
    elif n % 10 == 1:
        return f"{n}st"
    elif n % 10 == 2:
        return f"{n}nd"
    elif n % 10 == 3:
        return f"{n}rd"
    else:
        return f"{n}th"


def number_to_words(n):
    units = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ]
    teens = [
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    tens = [
        "",
        "ten",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    ]

    if n < 10:
        return units[n]
    elif n < 20:
        return teens[n - 10]
    else:
        if n % 10 == 0:
            return tens[n // 10]
        else:
            return tens[n // 10] + "-" + units[n % 10]


def ordinal_word(n):
    if n == 1:
        return "first"
    elif n == 2:
        return "second"
    elif n == 3:
        return "third"
    elif n < 20:
        return number_to_words(n) + "th"
    else:
        base_word = number_to_words(n)
        if n % 10 == 1:
            return base_word[:-2] + "first"
        elif n % 10 == 2:
            return base_word[:-2] + "second"
        elif n % 10 == 3:
            return base_word[:-2] + "third"
        else:
            return base_word + "th"


def get_paragraphs() -> List[str]:
    recipe_path = "datasets/test_set.json"
    with open(recipe_path, "r") as file:
        data = json.load(file)

    data["data"] = data["data"][:100]

    # Save the trimmed data to a new file
    with open("datasets/trimmed_recipes.json", "w") as file:
        json.dump(data, file)

    recipe_path = "datasets/trimmed_recipes.json"

    loader = JSONLoader(
        file_path=recipe_path,
        jq_schema="""
            .data[] | {
                "recipe_id": .recipe_id,
                "steps": [.context[] | {
                    "step_title": .title,
                    "instructions": .body
                }]
            }
        """,
        text_content=False,
        json_lines=True,
    )

    documents = loader.load()
    print("Number of documents (recipes):", len(documents))
    paragraphs = []
    with open("recipe_paragraphs.txt", mode="wt") as file:
        for doc_i, doc in enumerate(documents):
            # Access 'page_contents' which contains the needed information as a JSON string
            page_contents = doc.page_content
            if isinstance(page_contents, str):
                # Parse the JSON string into a dictionary if it's a string
                content_data = json.loads(page_contents)
            else:
                # Assume it's already a dictionary
                content_data = page_contents

            # Extracting the recipe ID and steps directly from the content data
            paragraph = ""
            recipe_id = content_data["recipe_id"]
            steps = content_data["steps"]
            paragraph += f"How to make {recipe_id}\n\n"
            for i, step in enumerate(steps):
                paragraph += f"The {ordinal_word(i+1)} step is: {step['step_title'].lower()}. {step['instructions']}\n\n"
            paragraph += "\n"
            paragraphs.append(paragraph)
            documents[doc_i].page_content = paragraph

        return paragraphs, documents


def update_index_store():
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")
    Settings.llm = None  # type: ignore
    paragraphs = get_paragraphs()
    documents = [Document(text=i) for i in paragraphs]

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    index.storage_context.persist(persist_dir="./storage_contexts/")
    print("Recipe size have been cut and vector store saved")

def index_to_faiss():
    paragraphs, documents = get_paragraphs()
    embeddings = OllamaEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("recipe_qa")

if __name__ == "__main__":
    download_dataset()
    index_to_faiss()
