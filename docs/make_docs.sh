doxygen Doxyfile

doxysphinx build ./ ./build_sphinx/html ./Doxyfile

sphinx-build -M html ./ ./build_sphinx

cp -r ./build_sphinx/html/* ./website
