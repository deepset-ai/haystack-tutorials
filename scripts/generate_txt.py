import argparse
import tomli
from nbconvert import MarkdownExporter
from nbconvert.filters.strings import get_lines
from subprocess import check_output

from pathlib import Path


def read_index(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def generate_metadata(config, tutorial):
    aliases = []
    if "aliases" in tutorial:
        for alias in tutorial["aliases"]:
            aliases.append(f"/tutorials/{alias}")

    last_commit_date = (
        check_output(f'git log -1 --pretty=format:"%cs" tutorials/{tutorial["notebook"]}'.split()).decode().strip()
    )

    return f"""layout: {config["layout"]}
featured: {tutorial.get("featured", False)}
colab: {tutorial.get("colab", f'{config["colab"]}{tutorial["notebook"]}')}
toc: {config["toc"]}
title: "{tutorial["title"]}"
lastmod: {last_commit_date}
level: "{tutorial["level"]}"
weight: {tutorial["weight"]}
description: {tutorial["description"]}
category: "QA"
aliases: {aliases}
download: "/downloads/{tutorial["notebook"]}"
completion_time: {tutorial.get("completion_time", False)}
created_at: {tutorial["created_at"]}
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
    parser.add_argument("--print-metadata", dest="metadata", action="store_true")
    args = parser.parse_args()
    index = read_index(args.index)

    notebooks = args.notebooks
    if args.notebooks == ["all"]:
        tutorials_path = Path(".", "tutorials")
        notebooks = tutorials_path.glob("[0-9]*.ipynb")

    notebooks_configs = {cfg["notebook"]: cfg for cfg in index["tutorial"]}

    for notebook in notebooks:
        notebook_name = notebook.split("/")[-1]
        tutorial_config = notebooks_configs.get(notebook_name)
        if tutorial_config:
            generate_markdown_from_notebook(tutorial_config, args.output, notebook)

            if args.metadata:
                meta = generate_metadata(index["config"], tutorial_config)
                print(meta)
