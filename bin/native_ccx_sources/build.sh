#!/bin/zsh
set -euo pipefail

# Rebuild the local macOS ARM CalculiX 2.23 binary from the supplied sources.
# Requires Homebrew gcc (gfortran), arpack, clang, make, and Python 3.
script_dir="${0:A:h}"
build_dir="${script_dir}/build"
mkdir -p "$build_dir"
cd "$build_dir"

python3 - "$script_dir" <<'PY'
import hashlib
import sys
from pathlib import Path
root = Path(sys.argv[1])
expected = {
    "ccx_2.23.src.tar.bz2": "9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7",
    "spooles.2.2.tgz": "a84559a0e987a1e423055ef4fdf3035d55b65bbe4bf915efaa1a35bef7f8c5dd",
}
for name, digest in expected.items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"SHA-256 mismatch: {name}")
PY

if [[ ! -d ccx_2.23/src ]]; then
  mkdir -p source
  tar -xjf "$script_dir/ccx_2.23.src.tar.bz2" -C source
  mv source/CalculiX/ccx_2.23 ./ccx_2.23
fi
if [[ ! -f spooles/Make.inc ]]; then
  mkdir -p spooles
  tar -xzf "$script_dir/spooles.2.2.tgz" -C spooles
fi

python3 - <<'PY'
from pathlib import Path
def replace(path, before, after):
    file = Path(path)
    source = file.read_text()
    if before in source:
        file.write_text(source.replace(before, after))
    elif after not in source:
        raise RuntimeError(f"Expected patch text absent: {path}: {before}")
replace("spooles/Make.inc", "/usr/lang-4.0/bin/cc", "clang")
replace("spooles/Tree/src/makeGlobalLib", "drawTree.c", "tree.c")
replace("spooles/ETree/src/transform.c", "IVinit(nfront, NULL)", "IVinit(nfront, 0)")
replace("ccx_2.23/src/Makefile", "ccx_2.23: $(OCCXMAIN) ccx_2.23.a  $(LIBS)",
        "ccx_2.23: $(OCCXMAIN) ccx_2.23.a")
file = Path("ccx_2.23/src/readnewmesh.c")
source = file.read_text()
before = "*iprfnp=iprfn;*konrfnp=konrfn;*ratiorfnp=ratiorfn;\n  \n  return NULL;"
after = "*iprfnp=iprfn;*konrfnp=konrfn;*ratiorfnp=ratiorfn;\n  \n  return;"
if before in source:
    file.write_text(source.replace(before, after, 1))
elif after not in source:
    raise RuntimeError("Expected readnewmesh.c patch text absent")
PY

make -C spooles lib > spooles_build.log 2>&1
make -C spooles/MT/src makeLib >> spooles_build.log 2>&1
make -C ccx_2.23/src ccx_2.23 \
  'CC=clang' 'FC=/opt/homebrew/bin/gfortran' \
  'CFLAGS=-O2 -I../../spooles -DARCH=Linux -DSPOOLES -DARPACK -DMATRIXSTORAGE -DUSE_MT=1' \
  'FFLAGS=-O2 -fopenmp -cpp' \
  'LIBS=../../spooles/spooles.a -L/opt/homebrew/opt/arpack/lib -larpack -framework Accelerate' \
  > ccx_build.log 2>&1
cp ccx_2.23/src/ccx_2.23 "$script_dir/../ccx_2.23"
echo "$script_dir/../ccx_2.23"
