set -e          # 任意命令返回非0时立即退出
set -u          # 使用未定义变量时报错
set -o pipefail # 管道中任意一个命令失败就算失败

SDK_VERSION=$(cat version)
echo "SDK_VERSION: $SDK_VERSION"


echo "Uninstall tactile_sdk ing..."
pip uninstall tactile_sdk -y || true    # 可选：卸载失败也允许继续（因为可能未安装）
sudo apt remove tactile_sdk -y || true  #同上


echo "Build tactile_sdk..."
# cmake -B build -GNinja -DBUILD_EXAMPLE=ON -DINFER_ENGINE=cuda -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake -B build -GNinja -DBUILD_PY=ON -DINFER_ENGINE=cuda -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build

cpack --config build/CPackConfig.cmake
sudo dpkg -i tactile_sdk-$SDK_VERSION-Linux.deb
cd py
pip install -e . --break-system-packages
cd ..
