@echo off
echo Compiling nearest neighbors...
cd utils\nearest_neighbors
python setup.py install --home="."
cd ..\..\

echo Compiling cpp subsampling...
cd utils\cpp_wrappers\cpp_subsampling
python setup.py build_ext --inplace
cd ..\..\..\

echo Compilation finished.
