#!/usr/bin/env bash
# #!/bin/bash

../simulation.py --chi 10 --eigen_index 2 --beta -2
../simulation.py --chi 10 --eigen_index 2 --beta -1
../simulation.py --chi 10 --eigen_index 2 --beta 0
../simulation.py --chi 10 --eigen_index 2 --beta 1
../simulation.py --chi 10 --eigen_index 2 --beta 2
../simulation.py --chi 10 --eigen_index 2 --beta 3
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --beta -2
../simulation.py --chi 30 --eigen_index 2 --beta -1
../simulation.py --chi 30 --eigen_index 2 --beta 0
../simulation.py --chi 30 --eigen_index 2 --beta 1
../simulation.py --chi 30 --eigen_index 2 --beta 2
../simulation.py --chi 30 --eigen_index 2 --beta 3
./genhtml.py

../simulation.py --chi 10 --eigen_index 2 --m -1
../simulation.py --chi 10 --eigen_index 2 --m -0.5
../simulation.py --chi 10 --eigen_index 2 --m 0
../simulation.py --chi 10 --eigen_index 2 --m 0.5
../simulation.py --chi 10 --eigen_index 2 --m 1
../simulation.py --chi 10 --eigen_index 2 --m 2
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --m -1
../simulation.py --chi 30 --eigen_index 2 --m -0.5
../simulation.py --chi 30 --eigen_index 2 --m 0
../simulation.py --chi 30 --eigen_index 2 --m 0.5
../simulation.py --chi 30 --eigen_index 2 --m 1
../simulation.py --chi 30 --eigen_index 2 --m 2
./genhtml.py

../simulation.py --chi 10 --eigen_index 2 --a -2
../simulation.py --chi 10 --eigen_index 2 --a -1
../simulation.py --chi 10 --eigen_index 2 --a 0
../simulation.py --chi 10 --eigen_index 2 --a 1
../simulation.py --chi 10 --eigen_index 2 --a 2
../simulation.py --chi 10 --eigen_index 2 --a 3
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --b -2
../simulation.py --chi 30 --eigen_index 2 --b -1
../simulation.py --chi 30 --eigen_index 2 --b 0
../simulation.py --chi 30 --eigen_index 2 --b 1
../simulation.py --chi 30 --eigen_index 2 --b 2
../simulation.py --chi 30 --eigen_index 2 --b 3
./genhtml.py

../simulation.py --chi 10 --eigen_index 2 --alpha -2
../simulation.py --chi 10 --eigen_index 2 --alpha -1
../simulation.py --chi 10 --eigen_index 2 --alpha 0
../simulation.py --chi 10 --eigen_index 2 --alpha 1
../simulation.py --chi 10 --eigen_index 2 --alpha 2
../simulation.py --chi 10 --eigen_index 2 --alpha 3
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --alpha -2
../simulation.py --chi 30 --eigen_index 2 --alpha -1
../simulation.py --chi 30 --eigen_index 2 --alpha 0
../simulation.py --chi 30 --eigen_index 2 --alpha 1
../simulation.py --chi 30 --eigen_index 2 --alpha 2
../simulation.py --chi 30 --eigen_index 2 --alpha 3
./genhtml.py

../simulation.py --chi 10 --eigen_index 2 --mu -2
../simulation.py --chi 10 --eigen_index 2 --mu -1
../simulation.py --chi 10 --eigen_index 2 --mu 0
../simulation.py --chi 10 --eigen_index 2 --mu 1
../simulation.py --chi 10 --eigen_index 2 --mu 2
../simulation.py --chi 10 --eigen_index 2 --mu 3
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --mu -2
../simulation.py --chi 30 --eigen_index 2 --mu -1
../simulation.py --chi 30 --eigen_index 2 --mu 0
../simulation.py --chi 30 --eigen_index 2 --mu 1
../simulation.py --chi 30 --eigen_index 2 --mu 2
../simulation.py --chi 30 --eigen_index 2 --mu 3
./genhtml.py

../simulation.py --chi 10 --eigen_index 2 --nu -2
../simulation.py --chi 10 --eigen_index 2 --nu -1
../simulation.py --chi 10 --eigen_index 2 --nu 0
../simulation.py --chi 10 --eigen_index 2 --nu 1
../simulation.py --chi 10 --eigen_index 2 --nu 2
../simulation.py --chi 10 --eigen_index 2 --nu 3
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --nu -2
../simulation.py --chi 30 --eigen_index 2 --nu -1
../simulation.py --chi 30 --eigen_index 2 --nu 0
../simulation.py --chi 30 --eigen_index 2 --nu 1
../simulation.py --chi 30 --eigen_index 2 --nu 2
../simulation.py --chi 30 --eigen_index 2 --nu 3
./genhtml.py

../simulation.py --chi 10 --eigen_index 2 --gamma -2
../simulation.py --chi 10 --eigen_index 2 --gamma -1
../simulation.py --chi 10 --eigen_index 2 --gamma 0
../simulation.py --chi 10 --eigen_index 2 --gamma 1
../simulation.py --chi 10 --eigen_index 2 --gamma 2
../simulation.py --chi 10 --eigen_index 2 --gamma 3
./genhtml.py

../simulation.py --chi 30 --eigen_index 2 --gamma -2
../simulation.py --chi 30 --eigen_index 2 --gamma -1
../simulation.py --chi 30 --eigen_index 2 --gamma 0
../simulation.py --chi 30 --eigen_index 2 --gamma 1
../simulation.py --chi 30 --eigen_index 2 --gamma 2
../simulation.py --chi 30 --eigen_index 2 --gamma 3
./genhtml.py

./upload.sh
