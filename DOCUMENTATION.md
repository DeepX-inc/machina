# How to create docmentation from docstrings in the codes
Install [Sphinx](http://www.sphinx-doc.org/ja/stable/index.html)
`pip install sphinx`
Create rst files.
Run the following command in the root directory of machina's repository.
* We ignore `setup.py`
`sphinx-apidoc -M -f -o ./docs . setup.py`
Build rst files and create html file.
Run the following command in the root directory of machina's repository.
`sphinx-build -b html ./docs ./docs/_build/`
Then, we can get the html file of documentation as `./docs/_build/index.html`.

# How to add documentation
If possible, additional docstring should be written in codes rather than in created rst files.
