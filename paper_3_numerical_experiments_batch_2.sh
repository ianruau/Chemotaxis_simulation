#!/usr/bin/env bash
# #!/bin/bash

# Non-minimal Supercritical case
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 47.8460 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_supercritical --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 47.9000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_supercritical --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 70.0000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_supercritical --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 150.000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_supercritical --save_data no

# Non-minimal Subcritical case
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 95.7600 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_subcritical --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 95.7660 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_subcritical --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 300.000 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_subcritical --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 600.000 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/non-minimal_subcritical --save_data no

# Minimal Supercritical case
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 43.4000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_supercritical --save_data no
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 43.8000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_supercritical --save_data no
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 100.000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_supercritical --save_data no
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 300.000 --time 1000 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_supercritical --save_data no

# Minimal Subcritical case
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 40 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 30 --eigen_mode_n 1 --chi 11941343161138.6 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_subcritical --save_data no
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 40 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 30 --eigen_mode_n 1 --chi 11941343161138.8 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_subcritical --save_data no
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 40 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 30 --eigen_mode_n 1 --chi 20000000000000.0 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_subcritical --save_data no
./simulation.py --u_star_fixed 1 --equilibrium_mode fixed --a 0 --b 0 --alpha 1 --m 1 --beta 40 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 30 --eigen_mode_n 1 --chi 50000000000000.0 --time 30 --save_static_plots no --output_dir paper_iii_numerical_experiments_batch_2/minimal_subcritical --save_data no
