# libcoral API docs

This directory holds the source files required to build the API reference for
the libcoral C++ library.

Essentially, you just need to run `makedocs.sh` to build everything.

Of course, it requires a few tool dependencies. So if it's your first time, then
you should set up as follows:

```
# We use Python3, so if that's not your default, start a virtual environment:
python3 -m venv ~/.my_venvs/coraldocs
source ~/.my_venvs/coraldocs/bin/activate

# Navigate to the libcoral/docs/ directory.

# Install doxygen:
sudo apt-get install doxygen

# Install other dependencies:
pip install -r requirements.txt

# Build the docs:
bash makedocs.sh
```

The results are output in `_build/`. The `_build/preview/` files are for local
viewing--just open the `index.html` page. The `_build/web/` files are designed
for publishing on the Coral website.

For more information about the syntax in the RST files, see the
[reStructuredText documentation](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
