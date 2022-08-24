#!/bin/bash
cd ..
find examples -name "* *" -type f | rename 's/ /_/g'
python -m nbconvert --to rst examples/*.ipynb --output-dir docs/examples
find examples -name "* *" -type f | rename 's/_/ /g'
cd docs