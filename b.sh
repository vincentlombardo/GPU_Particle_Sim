cd ~/particle-sim
rm -rf build
cmake -S . -B build
cmake --build build -j$(nproc)
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./build/particle-sim
