cd googletest        # Main directory of the cloned repository.
mkdir build          # Create a directory to hold the build output.
cd build
cmake -DCMAKE_INSTALL_PREFIX=$AMR_SOLVER/lib/install/googletest ..             # Generate native build scripts for GoogleTest.
make install
