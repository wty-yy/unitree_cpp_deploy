#!/usr/bin/env bash

set -Eeuo pipefail

camera_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${camera_dir}/build"
publisher_bin="${build_dir}/d435i/d435i_depth_publisher"
viewer_bin="${build_dir}/viewer/depth_camera_viewer"
publisher_pid=""
viewer_pid=""

usage() {
  cat <<'EOF'
Usage: ./deploy/camera/run.sh [--build]

  -b, --build  Configure and build before starting
  -h, --help   Show this help
EOF
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM

  for pid in "${viewer_pid}" "${publisher_pid}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${viewer_pid}" "${publisher_pid}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  exit "${status}"
}

build=false
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  case "$1" in
    -b|--build)
      build=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
fi

if [[ "${build}" == true ]]; then
  cmake -S "${camera_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${build_dir}" -j
fi

if [[ ! -x "${publisher_bin}" || ! -x "${viewer_bin}" ]]; then
  echo "Camera binaries not found. Run ./deploy/camera/run.sh --build first." >&2
  exit 1
fi

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stdbuf -oL -eL "${publisher_bin}" \
  > >(sed -u 's/^/[d435i] /') \
  2> >(sed -u 's/^/[d435i] /' >&2) &
publisher_pid=$!

# Let the RealSense stream and DDS writer initialize before creating readers.
sleep 2
if ! kill -0 "${publisher_pid}" 2>/dev/null; then
  set +e
  wait "${publisher_pid}"
  status=$?
  set -e
  exit "${status}"
fi

viewer_stdin="/dev/null"
if [[ -t 0 ]]; then
  viewer_stdin="/dev/tty"
fi
stdbuf -oL -eL "${viewer_bin}" < "${viewer_stdin}" \
  > >(sed -u 's/^/[viewer] /') \
  2> >(sed -u 's/^/[viewer] /' >&2) &
viewer_pid=$!

set +e
wait -n "${publisher_pid}" "${viewer_pid}"
status=$?
set -e
exit "${status}"
