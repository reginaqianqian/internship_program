# tactile_sdk

## Overview
SDK for sharpa tactile sensor

## Build Instructions

build options:
- BUILD_PY ['ON', 'OFF']
- BUILD_EXAMPLE ['ON', 'OFF']
- BUILD_TEST ['ON', 'OFF']
- INFER_ENGINE ['none', 'cuda', 'coreml']

e.g. build with python binding and cuda

```
cmake -B build -GNinja -DBUILD_PY=ON -DINFER_ENGINE=cuda -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

install
```
cpack --config build/CPackConfig.cmake
sudo dpkg -i tactile_sdk-{your version}-Linux.deb
cd py
pip install -e .
```

cpp example (BUILD_EXAMPLE have to be set to 'ON')
```
./build/example_cpp/example-fetch-raw
./build/example_cpp/example-fetch-raw-callback
./build/example_cpp/example-infer
```

py example (BUILD_PY have to be set to 'ON')
```
python example_py/example_fetch_raw.py
python example_py/example_fetch_raw_callback.py
python example_py/example_infer.py --cfg config/cfg.py
python example_py/example_infer_callback.py --cfg config/cfg.py
```

uninstall
```
pip uninstall tactile_sdk
sudo apt remove tactile_sdk
```