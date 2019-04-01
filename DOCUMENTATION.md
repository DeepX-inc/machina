# How to improve the documentation
## Prepare
Install [Sphinx](http://www.sphinx-doc.org/ja/stable/index.html)
`pip install sphinx`
## Build
Build rst files and create html file.
Run the following command in the root directory of machina's repository.
`sphinx-build -b html ./docs/source ./docs/_build/`
Then, we can get the html file of documentation as `./docs/_build/index.html`.
## Publish
Pubulish `index.html`
Run the following command.
`mv docs/_build/index.html docs/`
Then, we can see updated html in this [url](https://deepx.github.io/machina/).

If possible, additional docstring should be written in codes rather than in created rst files.
