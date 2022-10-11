import argparse
from datetime import date
import tomli
from nbconvert import MarkdownExporter

def read_index(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def generate_frontmatter(config, tutorial):
    aliases = []
    if "aliases" in tutorial:
        for alias in tutorial["aliases"]:
            aliases.append(f"/tutorials/{alias}")

    frontmatter = f"""---
layout: {config["layout"]}
colab: {config["colab"]}{tutorial["notebook"]}
toc: {config["toc"]}
title: "{tutorial["title"]}"
last_updated: {date.today()}
level: "{tutorial["level"]}"
weight: {tutorial["weight"]}
description: {tutorial["description"]}
category: "QA"
aliases: {aliases}
---
    """
    return frontmatter


def generate_markdown_from_notebook(config, tutorial, output_path, tutorials_path):
    frontmatter = generate_frontmatter(config, tutorial)
    md_exporter = MarkdownExporter(exclude_output=True)
    body, _ = md_exporter.from_filename(f"{tutorials_path}")
    print(f"Processing {tutorials_path}")

    with open(f"{output_path}/{tutorial['notebook'][:-6]}.md", "w", encoding="utf-8") as f:
        try:
            f.write(frontmatter + "\n\n")
        except IndexError as e:
            raise IndexError(
                "Can't find the header for this tutorial. Have you added it in 'scripts/generate_markdowns.py'?"
            ) from e
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", dest="index")
    parser.add_argument("--notebooks", dest="notebooks", nargs='+', default=[])
    parser.add_argument("--output", dest="output", default="markdowns")
    args = parser.parse_args()
    index = read_index(args.index)

    for notebook in args.notebooks:
        for tutorial in index["tutorial"]:
            if  tutorial["notebook"] in notebook:
                generate_markdown_from_notebook(index["config"], tutorial, args.output, notebook)