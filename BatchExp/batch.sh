#!/usr/bin/env bash
# #!/bin/bash

../simulation.py --confirm no --chi 70 --EigenIndex 2
../simulation.py --confirm no --chi 70 --EigenIndex 3
../simulation.py --confirm no --chi 70 --EigenIndex 4
../simulation.py --confirm no --chi 70 --EigenIndex 5
# ../simulation.py --confirm no --chi 70 --EigenIndex 6
# ../simulation.py --confirm no --chi 70 --EigenIndex 7
./genhtml.py

../simulation.py --confirm no --chi 150 --EigenIndex 2
../simulation.py --confirm no --chi 150 --EigenIndex 3
../simulation.py --confirm no --chi 150 --EigenIndex 4
# ../simulation.py --confirm no --chi 150 --EigenIndex 5
# ../simulation.py --confirm no --chi 150 --EigenIndex 6
# ../simulation.py --confirm no --chi 150 --EigenIndex 7
./genhtml.py

../simulation.py --confirm no --chi 300 --EigenIndex 2
../simulation.py --confirm no --chi 300 --EigenIndex 3
../simulation.py --confirm no --chi 300 --EigenIndex 4
# ../simulation.py --confirm no --chi 300 --EigenIndex 5
# ../simulation.py --confirm no --chi 300 --EigenIndex 6
# ../simulation.py --confirm no --chi 300 --EigenIndex 7
./genhtml.py

../simulation.py --confirm no --chi 490 --EigenIndex 2
../simulation.py --confirm no --chi 490 --EigenIndex 3
../simulation.py --confirm no --chi 490 --EigenIndex 4
# ../simulation.py --confirm no --chi 490 --EigenIndex 5
# ../simulation.py --confirm no --chi 490 --EigenIndex 6
# ../simulation.py --confirm no --chi 490 --EigenIndex 7
./genhtml.py

../simulation.py --confirm no --chi 700 --EigenIndex 2
../simulation.py --confirm no --chi 700 --EigenIndex 3
# ../simulation.py --confirm no --chi 700 --EigenIndex 4
# ../simulation.py --confirm no --chi 700 --EigenIndex 5
# ../simulation.py --confirm no --chi 700 --EigenIndex 6
# ../simulation.py --confirm no --chi 700 --EigenIndex 7
./genhtml.py

./upload.sh
