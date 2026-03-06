if [ -z "$1" ]; then
    echo "dir_bin not provided."
    exit 1
fi

dir_bin=$1
dir_module='tactile_sdk/sharpa/tactile'

lib_suffix='cpython-310-x86_64-linux-gnu'

rm $dir_module/*.so
rm -rf build
# cp $dir_bin/tactile_sdk_py.$lib_suffix.so $dir_module/tactile_sdk.$lib_suffix.so
cp $dir_bin/*.so $dir_module
 
if command -v python3 &> /dev/null; then
    python3 setup.py bdist_wheel
else
    echo "python3 could not be found, using python"
    python setup.py bdist_wheel
fi