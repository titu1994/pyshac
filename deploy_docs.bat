pip install -e .
cd docs
python autogen.py
cd ..
mkdocs build
rmdir /S /Q "../titu1994.github.io/pyshac/"
xcopy "site" "../titu1994.github.io/pyshac" /E /H /S
rmdir /S /Q "build"
rmdir /S /Q "pyshac.egg-info"
rmdir /S /Q site