# Local CalculiX build

`../ccx_2.23` is the native macOS ARM executable used in the recorded runs.
Its SHA-256 is
`d4002f6eb0f021790d1429cf08088f29e7f56a97aea57179e4f4e9da4c80bbf3`.
It links to Homebrew ARPACK and GCC runtime libraries and Apple's Accelerate
framework. Gmsh 4.15.2 generated the recorded meshes.

The supplied CalculiX 2.23 and SPOOLES 2.2 source archives are the exact
archives used for the build. `build.sh` verifies their hashes, applies the
macOS compiler fixes, and rebuilds the executable. Install Homebrew
`gcc` and `arpack`, then run `./build.sh` in this directory. The build
sources and their own notices govern reuse of the solver and its libraries.
