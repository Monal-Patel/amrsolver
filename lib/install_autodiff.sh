cd autodiff
rm -rf .build
mkdir .build && cd .build
cmake .. -DCMAKE_PREFIX_PATH=/home/monal/.local/lib/python3.10/site-packages -DAUTODIFF_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$AMR_SOLVER/lib/install/autodiff
cmake --build . --target install -j 4
