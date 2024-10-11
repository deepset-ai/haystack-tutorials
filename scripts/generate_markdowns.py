import argparse
import tomli
from nbconvert import MarkdownExporter
from nbconvert.filters.strings import get_lines
from subprocess import check_output


def read_index(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def generate_frontmatter(config, tutorial):
    aliases = []
    if "aliases" in tutorial:
        for alias in tutorial["aliases"]:
            aliases.append(f"/tutorials/{alias}")

    last_commit_date = (
        check_output(f'git log -1 --pretty=format:"%cs" tutorials/{tutorial["notebook"]}'.split()).decode().strip()
    )

    frontmatter = f"""---
layout: {config["layout"]}
featured: {tutorial.get("featured", False)}
featured_image: /images/tutorial_walkthrough_thumbnail.png
images: ["/images/tutorial_walkthrough_thumbnail.png"]
haystack_version: "{tutorial.get("haystack_version", "latest")}"
haystack_2: {tutorial.get("haystack_2", False)}
hidden: {tutorial.get("hidden", False)}
sitemap_exclude: {tutorial.get("sitemap_exclude", False)}
colab: {tutorial.get("colab", f'{config["colab"]}{tutorial["notebook"]}')}
toc: {config["toc"]}
title: "{tutorial["title"]}"
lastmod: {last_commit_date}
level: "{tutorial.get("level", False)}"
weight: {tutorial["weight"]}
description: {tutorial["description"]}
category: "QA"
aliases: {aliases}
download: {tutorial.get("download", f'/downloads/{tutorial["notebook"]}')}
completion_time: {tutorial.get("completion_time", False)}
created_at: {tutorial.get("created_at", False)}
guide: {tutorial.get("guide", False)}
---
    """
    return frontmatter


def generate_markdown_from_notebook(config, tutorial, output_path, tutorials_path):
    frontmatter = generate_frontmatter(config, tutorial)
    md_exporter = MarkdownExporter(exclude_output=True)
    body, _ = md_exporter.from_filename(f"{tutorials_path}")
    body = get_lines(body, start=1)
    print(f"Processing {tutorials_path}")
    filename = tutorial.get("slug", tutorial["notebook"][:-6])
    with open(f"{output_path}/{filename}.md", "w", encoding="utf-8") as f:
        try:
            f.write(frontmatter + "\n\n")
        except IndexError as e:
            raise IndexError(
                "Can't find the header for this tutorial. Have you added it in 'scripts/generate_markdowns.py'?"
            ) from e
        f.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", dest="index", default="index.toml")
    parser.add_argument("--notebooks", dest="notebooks", nargs="+", default=[])
    parser.add_argument("--output", dest="output", default="markdowns")
    args = parser.parse_args()
    index = read_index(args.index)

    if args.notebooks == ["all"]:
        for config in index["tutorial"]:
            notebook = "tutorials/" + config["notebook"]
            print(notebook)
            generate_markdown_from_notebook(index["config"], config, args.output, notebook)

    else:
        nb_to_config = {cfg["notebook"]: cfg for cfg in index["tutorial"]}

        for notebook in args.notebooks:
            nb_name = notebook.split("/")[-1]
            tutorial_cfg = nb_to_config.get(nb_name)
            if tutorial_cfg:
                generate_markdown_from_notebook(index["config"], tutorial_cfg, args.output, notebook)
