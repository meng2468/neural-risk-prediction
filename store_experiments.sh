#!/bin/bash

echo "Moving all experiment files into: $1";

mkdir -p  "evaluation/$1";

mv evaluation/*.csv evaluation/$1/
mv evaluation/*.jpeg evaluation/$1/

