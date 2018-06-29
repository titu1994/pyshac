pip install --upgrade pyshac
cd docs
python autogen.py
cd ..
mkdocs build
rmdir /S "../titu1994.github.io/pyshac/"
xcopy "site" "../titu1994.github.io/pyshac" /E /H
