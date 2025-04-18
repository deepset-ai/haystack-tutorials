## This script is used to create a metrix to test tutorials

import argparse
import json

import tomllib


def read_index(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="""python generate_matrix.py --haystack-version v2.10.0""")
    parser.add_argument("--index", dest="index", default="index.toml")
    parser.add_argument("--notebooks", dest="notebooks", nargs="+", default=[])
    parser.add_argument("--haystack-version", dest="version", required=True)
    parser.add_argument("--include-main", dest="main", action="store_true")

    args = parser.parse_args()
    index = read_index(args.index)

    matrix = []
    for tutorial in index["tutorial"]:
        if tutorial.get("hidden", False):
            # Do not waste CI time on hidden tutorials
            continue

        if tutorial.get("needs_gpu", False):
            # We're not running GPU tutorials on GitHub Actions
            # since we don't have a GPUs there
            continue

        if not tutorial.get("colab", True):
            # This tutorial doesn't have any runnable Python code
            # so there's nothing to test
            continue

        notebook = tutorial["notebook"]
        if args.notebooks and notebook not in args.notebooks:
            # If the user specified a list of notebooks to run, only run those
            # otherwise run all of them
            continue

        version = tutorial.get("haystack_version", args.version)
        if version[0] != "v":
            version = f"v{version}"

        matrix.append(
            {"notebook": notebook[:-6], "haystack_version": version, "dependencies": tutorial.get("dependencies", [])}
        )

        if args.main and "haystack_version" not in tutorial:
            # If a tutorial doesn't specify a version, we also test it on main
            matrix.append(
                {
                    "notebook": notebook[:-6],
                    "haystack_version": "main",
                    "dependencies": tutorial.get("dependencies", []),
                }
            )

    print(json.dumps(matrix))
