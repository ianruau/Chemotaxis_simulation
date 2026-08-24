#!/usr/bin/env bash
# #!/bin/bash

#Supercritical case
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 30 --eigen_mode_n 1 --chi 47.8460 --time 500 --save_static_plots no --output_dir section_6-4 --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 47.9000 --time 1000 --save_static_plots no --output_dir section_6-4 --save_data no

#Subcritical case
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 95.7600 --time 30 --save_static_plots no --output_dir section_6-4 --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 95.7660 --time 30 --save_static_plots no --output_dir section_6-4 --save_data no
