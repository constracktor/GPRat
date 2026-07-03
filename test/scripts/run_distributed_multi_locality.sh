#!/usr/bin/env bash
# Smoke-tests the gprat_distributed binary across N HPX localities on one node.
# Usage: run_distributed_multi_locality.sh <path-to-gprat_distributed> <N>
#
# This only exercises multi-locality startup/teardown and one small correctness run; it requires
# a binary built against an HPX with networking enabled (networking=none rejects --hpx:localities
# outright), which is why the CTest entries that invoke this script are opt-in via
# GPRAT_TEST_MULTI_LOCALITY (see test/CMakeLists.txt).
set -e

BIN="$1"
N="$2"

if [[ -z "$BIN" || -z "$N" ]]; then
  echo "usage: $0 <path-to-gprat_distributed> <N>" 1>&2
  exit 1
fi

# HPX's TCP parcelport zero-copy path hangs once tile sizes exceed the default 8192-byte
# threshold in a multi-locality run; raise it to avoid this (see top-level README).
ZC_ARGS=(--hpx:ini=hpx.parcel.zero_copy_serialization_threshold=999999999)

pids=()

"$BIN" --hpx:localities="$N" --hpx:node=0 "${ZC_ARGS[@]}" \
  --start 128 --end 128 --step 2 --tiles 2 --loop 1 --output_csv /dev/null &
pids+=($!)

for ((node = 1; node < N; node++)); do
  "$BIN" --hpx:localities="$N" --hpx:node="$node" "${ZC_ARGS[@]}" &
  pids+=($!)
done

# `wait pid1 pid2 ...` only reports the last PID's status, so check each individually to make
# sure a failure on any locality fails the test.
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

exit "$failed"
