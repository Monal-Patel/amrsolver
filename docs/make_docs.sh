rm -rv build_doxygen
rm -rv website/*
doxygen Doxyfile
doxysphinx build ./ ./website ./Doxyfile
sphinx-build -b html ./ ./website