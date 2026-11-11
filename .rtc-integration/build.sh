#!/bin/bash
# Build the rumoca command line tool of the rtc-integration branch: target/release/rumoca.
# rtc-tools finds it through RTCTOOLS_RUMOCA.
set -euxo pipefail
cd "$(dirname "$0")/.."
cargo build --release --bin rumoca
