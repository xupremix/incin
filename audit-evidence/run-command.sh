#!/usr/bin/env bash
set -u

if [ "$#" -lt 3 ]; then
    echo "usage: $0 TASK-ID OUTPUT-FILE COMMAND..." >&2
    exit 2
fi

evidence_task=$1
evidence_output=$2
shift 2
evidence_command=$*
evidence_root="audit-evidence/${evidence_task}"
evidence_result="${evidence_root}/test-results/${evidence_output}"

mkdir -p "${evidence_root}/test-results"

evidence_start=$(date --iso-8601=seconds)
evidence_commit=$(git rev-parse HEAD)
evidence_cwd=$(pwd)

{
    echo "COMMAND: ${evidence_command}"
    echo "WORKING DIRECTORY: ${evidence_cwd}"
    echo "FEATURES/ENVIRONMENT: RUSTFLAGS=${RUSTFLAGS-<unset>} RUSTDOCFLAGS=${RUSTDOCFLAGS-<unset>} TRYBUILD=${TRYBUILD-<unset>} INCIN_DOCS=${INCIN_DOCS-<unset>}"
    echo "START: ${evidence_start}"
    echo "COMMIT: ${evidence_commit}"
    echo "FULL OUTPUT: ${evidence_result}"
} >> "${evidence_root}/commands.log"

set +e
bash -lc "${evidence_command}" > >(tee "${evidence_result}") 2>&1
evidence_exit=$?
set -e

evidence_end=$(date --iso-8601=seconds)
{
    echo "END: ${evidence_end}"
    echo "EXIT CODE: ${evidence_exit}"
    echo
} >> "${evidence_root}/commands.log"

exit "${evidence_exit}"
