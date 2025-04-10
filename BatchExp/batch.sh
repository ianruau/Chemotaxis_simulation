#!/usr/bin/env bash
# #!/bin/bash

../simulation.py --chi 10 --EigenIndex 2
../simulation.py --chi 10 --EigenIndex 3
../simulation.py --chi 10 --EigenIndex 4
../simulation.py --chi 10 --EigenIndex 5
../simulation.py --chi 10 --EigenIndex 6
../simulation.py --chi 10 --EigenIndex 7
./genhtml.py

../simulation.py --chi 23 --EigenIndex 2
../simulation.py --chi 23 --EigenIndex 3
../simulation.py --chi 23 --EigenIndex 4
../simulation.py --chi 23 --EigenIndex 5
../simulation.py --chi 23 --EigenIndex 6
../simulation.py --chi 23 --EigenIndex 7
./genhtml.py

../simulation.py --chi 24 --EigenIndex 2
../simulation.py --chi 24 --EigenIndex 3
../simulation.py --chi 24 --EigenIndex 4
../simulation.py --chi 24 --EigenIndex 5
../simulation.py --chi 24 --EigenIndex 6
../simulation.py --chi 24 --EigenIndex 7
./genhtml.py

../simulation.py --chi 70 --EigenIndex 2
../simulation.py --chi 70 --EigenIndex 3
../simulation.py --chi 70 --EigenIndex 4
../simulation.py --chi 70 --EigenIndex 5
../simulation.py --chi 70 --EigenIndex 6
../simulation.py --chi 70 --EigenIndex 7
./genhtml.py

../simulation.py --chi 150 --EigenIndex 2
../simulation.py --chi 150 --EigenIndex 3
../simulation.py --chi 150 --EigenIndex 4
../simulation.py --chi 150 --EigenIndex 5
../simulation.py --chi 150 --EigenIndex 6
../simulation.py --chi 150 --EigenIndex 7
./genhtml.py

../simulation.py --chi 300 --EigenIndex 2
../simulation.py --chi 300 --EigenIndex 3
../simulation.py --chi 300 --EigenIndex 4
../simulation.py --chi 300 --EigenIndex 5
../simulation.py --chi 300 --EigenIndex 6
../simulation.py --chi 300 --EigenIndex 7
./genhtml.py

../simulation.py --chi 490 --EigenIndex 2
../simulation.py --chi 490 --EigenIndex 3
../simulation.py --chi 490 --EigenIndex 4
../simulation.py --chi 490 --EigenIndex 5
../simulation.py --chi 490 --EigenIndex 6
../simulation.py --chi 490 --EigenIndex 7
./genhtml.py

../simulation.py --chi 700 --EigenIndex 2
../simulation.py --chi 700 --EigenIndex 3
../simulation.py --chi 700 --EigenIndex 4
../simulation.py --chi 700 --EigenIndex 5
../simulation.py --chi 700 --EigenIndex 6
../simulation.py --chi 700 --EigenIndex 7
./genhtml.py

./upload.sh
