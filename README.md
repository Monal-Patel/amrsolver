# amrsolver

## docs 
[Documentation](https://monal-patel.github.io/amrsolver_docs/)
## src
### cls
Closures (thermodynamic and transport)

### ibm
Immersed boundaries

### set
AMR initialisation and boundary conditions

### rhs
Fluxes and source term discretisation methods
- Compressible
- Incompressible

### tim
Time integration method


## Running
### Without IBM (with CUDA)
1. Install CUDA and ensure relevant environment variables are set in your shell, as per the standard CUDA installation guide.
2. Set `AMREX_HOME` variable in your shell environment to the amrex submodule(`./lib/amrex`) path.
3. `make` an executable using existing examples in `exm/"example_name"` folders. For example, "example_name" can be `Sods`.
4. Run with `./"executable_name" inputs` in `exm` folder.
