cd googletest        # Main directory of the cloned repository.
mkdir build          # Create a directory to hold the build output.
cd build
cmake -Dgtest_build_samples=ON -DCMAKE_INSTALL_PREFIX=$AMREX_HOME/../install/googletest ..             # Generate native build scripts for GoogleTest.
make install
