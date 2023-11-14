import argparse
import tomli
from nbconvert import MarkdownExporter
from nbconvert.filters.strings import get_lines
from subprocess import check_output

from pathlib import Path


def read_index(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def generate_metadata(tutorial):
    file_name = tutorial["notebook"].split(".")[0].lower()
    slug = tutorial.get("slug", f"tutorials/{file_name}")

    return f"""featured: {tutorial.get("featured", False)}
title: "{tutorial["title"]}"
haystack_version: "{tutorial.get("haystack_version", "latest")}"
level: "{tutorial["level"]}"
description: {tutorial["description"]}
completion_time: "{tutorial.get("completion_time", "")}"
link: {slug}
"""


def generate_markdown_from_notebook(tutorial, output_path, tutorials_path):
    md_exporter = MarkdownExporter(exclude_output=True)
    body, _ = md_exporter.from_filename(f"{tutorials_path}")
    body = get_lines(body, start=1)
    filename = tutorial.get("slug", tutorial["notebook"][:-6])
    Path(output_path).mkdir(exist_ok=True)
    with open(f"{output_path}/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", dest="index", default="index.toml")
    parser.add_argument("--notebooks", dest="notebooks", nargs="+", default=[])
    parser.add_argument("--output", dest="output", default="text")
    parser.add_argument("--metadata", dest="metadata", action="store_true")
    args = parser.parse_args()
    index = read_index(args.index)

    notebooks = args.notebooks
    if args.notebooks == ["all"]:
        tutorials_path = Path(".", "tutorials")
        notebooks = tutorials_path.glob("[0-9]*.ipynb")

    notebooks_configs = {cfg["notebook"]: cfg for cfg in index["tutorial"]}
    # print(notebooks_configs)

    for notebook in notebooks:
        notebook_name = str(notebook).split("/")[-1]
        tutorial_config = notebooks_configs.get(notebook_name)
        if tutorial_config and not tutorial_config.get("hidden", False):
            generate_markdown_from_notebook(tutorial_config, args.output, notebook)
            print(tutorial_config, "\n")

            if args.metadata:
                meta = generate_metadata(tutorial_config)
                meta_file_name = f"{notebook_name.split('.')[0]}.yml"
                Path(args.output, meta_file_name).write_text(meta)
