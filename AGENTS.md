# AGENTS.md

Instructions for AI coding agents (Claude Code, Copilot, Cursor, etc.) working in this repository.

## What this repo is

The tutorials shown on the [Haystack website](https://haystack.deepset.ai/tutorials/). Each tutorial is
an interactive `.ipynb` notebook that teaches a Haystack feature, pipeline pattern, or best practice —
not a showcase for a third-party product.

## Contribution process — read before writing a new tutorial

New tutorials require an issue **assigned by a maintainer** before a PR is opened. Do not write a new
notebook and open a PR for it speculatively — a CI workflow
(`.github/workflows/enforce_issue_link.yml`) auto-closes PRs that add a new tutorial without
referencing an issue assigned to the PR author. Full process in [CONTRIBUTING.md](CONTRIBUTING.md).

This gate only applies to PRs that add a **new** tutorial notebook. Fixes, edits, and content
improvements to existing tutorials don't need a pre-assigned issue.

If you're an agent acting on behalf of a user who wants to add a new tutorial:
1. Check whether an issue for it already exists and is assigned to the user.
2. If not, tell the user to open one via `.github/ISSUE_TEMPLATE/new_tutorial.yml` and wait for
   assignment.
3. Only write the notebook and PR once the issue is confirmed assigned.

## Adding/editing a tutorial

1. Copy [tutorials/template.ipynb](tutorials/template.ipynb) to start a new tutorial.
2. Name the file following the [naming convention](CONTRIBUTING.md#naming-convention-for-file-names):
   number prefix, underscores between words, short descriptive name.
3. Register it in `index.toml` under `[[tutorial]]` with `title`, `description`, `level`, `weight`,
   and `notebook`. `weight` controls ordering. Set `colab = false` if it can't run on Google Colab.
4. Update `README.md`'s tutorial table if adding a new tutorial.
5. Install pre-commit hooks (`pre-commit install`) so formatting checks run before you commit.

## Don't

- Don't add a tutorial that primarily promotes a third-party tool with Haystack as an afterthought.
- Don't open a PR adding a new tutorial without a linked, assigned issue — it will be auto-closed.
- Don't hardcode API keys or secrets in notebook cells or outputs.
