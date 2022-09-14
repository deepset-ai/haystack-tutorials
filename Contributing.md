# Contributing to Haystack Tutorials

All of the Haystack tutorials live in the `tutorials` folder in this repo. Each tutorial is an interactive `.ipynb` file that can be ran on Google Colab too. Follow the steps below to add or edit the tutorials 👇

1. Prepare your environment:
   - Install requirements with `pip install requirements.txt`
   - Install the pre-commit hooks with `pre-commit install`. This utility will run some formatting/checking
   tasks right before all git commit operations.
2. Please make sure to follow the [naming convention](#naming-convention) for file names.
3. Make any changes (editing an existing tutorial or creating a new one) in the `/tutorials` folder by editing or creating `.ipynb` files.
4. Update the `markdowns` folder to reflect the changes:
    - Run `python /scripts/generate_markdowns.py`
    - This will generate or update the relevant markdown file in `/markdowns`
5. Create a Pull Request
6. Wait for the [CI](#ci-continuous-integration) checks to pass
    - These checks will pass if the relevant markdown files have been created
7. Update the [README](./README.md) if necessary.
8. Wait for a review and merge 🎉 Thank you for contributing 💙


# CI (Continuous Integration)

We use GitHub Action for our Continuous Integration tasks. This means that, as soon as you open a PR, GitHub will start executing some workflows on your code. Here, the workflow will check that you've generated the required `.md` file for the tutorial(s) you've edited or created.

If all goes well, at the bottom of your PR page you should see something like this, where all checks are green.

![](https://raw.githubusercontent.com/deepset-ai/haystack/main/docs/img/ci-success.png)

If you see some red checks, then something didn't work, and action is needed from your side. This is most likely because you haven't updated or created the required `.md` file to accompany the tutorial that you've created or changed. 

# Naming Convention

- Each tutorial starts with its number
- Separate words with an `_` underscore
- Use a short descriptive name for the filename
- Generated markdown files only have the number of the tutorial (use the `scripts/generate_markdowns.py` script for this)