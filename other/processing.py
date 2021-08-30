import json
import os
from collections import defaultdict
from pathlib import Path


def get_files_path(folder_root, folder_list):
    """This method takes all json files and return a file names dict:
    {'folder': [path_file_1, path_file_2, ...].
    """
    path_dict = {}
    for folder in folder_list:
        path_dict[folder] = [
            os.path.join(path, name)
            for path, subdirs, files in os.walk(os.path.join(folder_root, folder))
            for name in files
            if not name.startswith(".")
        ]
    return path_dict


def get_text(file_list, folder_save, lang_threshold):
    """Takes all JSON files from folder and
    reconstructs initial text to text chunks with coordinates.
    This method saves processed JSON files from a folder to separate files.

    Args:
        file_list: files list for processing
        folder_save: folder for save already processed files
        lang_threshold: threshold for English language
    """
    if len(file_list) > 0:
        folder = file_list[0].split("/")[-3]

        doc_dict = defaultdict(dict)
        for file_path in file_list:

            with open(file_path) as f:
                data = json.load(f)

            doc_name = data["inputConfig"]["gcsSource"]["uri"]
            for i in range(len(data["responses"])):
                response = data["responses"][i]
                page_number = response["context"]["pageNumber"]
                text_blocks = []
                if "fullTextAnnotation" in response.keys():
                    for page in response["fullTextAnnotation"]["pages"]:
                        for block in page["blocks"]:
                            if "property" in block.keys():
                                lang_dic = {
                                    lang["languageCode"]: lang["confidence"]
                                    for lang in block["property"]["detectedLanguages"]
                                }
                                if lang_dic.get("en", 0) >= lang_threshold:
                                    text = ""
                                    for paragraph in block["paragraphs"]:
                                        for word in paragraph["words"]:
                                            text = (
                                                text
                                                + "".join(
                                                    [
                                                        symbol["text"]
                                                        for symbol in word["symbols"]
                                                    ]
                                                )
                                                + " "
                                            )
                                    if len(text) != 0:
                                        coordinate = [
                                            (c.get("x", 0), c.get("y", 0))
                                            for c in block["boundingBox"][
                                                "normalizedVertices"
                                            ]
                                        ]
                                        text_blocks.append((text, coordinate))
                doc_dict[doc_name]["page_{}".format(page_number)] = text_blocks

        Path(folder_save).mkdir(parents=True, exist_ok=True)
        with open("{}/{}.json".format(folder_save, folder), "w") as fp:
            json.dump(doc_dict, fp)
