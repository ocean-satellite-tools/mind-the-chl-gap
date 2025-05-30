# Jupyter Book

Everything is here.

`_config.yml` is set to not render the notebooks. So make sure to save in rendered format.

### GitHub Action

There is a GitHub Action that should build the book whenever there is a push to the `book` directory. If the Action does not run, then you will need to debug. Click on the Action that did not build and click on the part that had a problem.

### Build locally and push to GitHub

Do `pip install ghp-import` if needed. Then build book and push to GitHub. Set Pages to use `gh-pages` branch. These commands are run within the `book` directory.

```
cd /book
jupyter-book build . --keep-going
ghp-import -n -p -f _build/html
```

### Building Locally

1. Open a terminal.
2. Run `jupyter-book clean book/` to remove any existing builds
3. Run `jupyter-book build book/`

A fully-rendered HTML version of the book will be built in `book/_build/html/`.


