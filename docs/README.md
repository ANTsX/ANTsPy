# Building the docs

An action builds the docs on PRs. To update search, install sphinx and generate stubs with

```
sphinx-autogen -o docs/source/api docs/source/api_index.rst
```

from the ANTsPy/ directory.
