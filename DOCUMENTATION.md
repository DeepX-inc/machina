# How to improve the documentation

## Add new contents
If possible, additional docstring should be written inside codes rather than in rst files.

## Prepare
Install [Sphinx](http://www.sphinx-doc.org/ja/stable/index.html)
`pip install sphinx`

## Build
Build rst files and create html files, css files and so on.
Run the following command in the root directory of machina's repository.
`sphinx-build -b html ./docs/source ./docs/_build/`

## Publish
Pubulish 
Run the following command.
`cp -r docs/_build/ docs/`
Then, we can see updated html in this [url](https://deepx-inc.github.io/machina/).
