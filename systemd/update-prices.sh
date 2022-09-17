#!/bin/bash

pushd ~/espial/money/feevee

source env/bin/activate

./run-cli --prices --daily

popd
