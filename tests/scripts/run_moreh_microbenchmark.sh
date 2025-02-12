#!/bin/bash
# set -x
set -eo pipefail

run_profiling_test() {
  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ "$TT_METAL_DEVICE_PROFILER" != 1 ]]; then
    echo "Must set TT_METAL_DEVICE_PROFILER to 1 to run microbenchmarks" 1>&2
    exit 1
  fi

  echo "Make sure this test runs in a build with ENABLE_PROFILER=1"

  source build/python_env/bin/activate
  export PYTHONPATH=$TT_METAL_HOME

    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_h2d_dram
    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_d2h_dram
    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_h2d_l1 -k $ARCH_NAME
    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_pcie_d2h_l1 -k $ARCH_NAME
    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_noc -k $ARCH_NAME
    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_matmul_dram -k $ARCH_NAME
    pytest $TT_METAL_HOME/tests/scripts/test_moreh_microbenchmark.py::test_matmul_l1 -k $ARCH_NAME
}

run_profiling_test
