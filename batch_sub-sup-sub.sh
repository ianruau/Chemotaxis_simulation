#!/usr/bin/env bash
# #!/bin/bash

#Supercritical case
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 30 --eigen_mode_n 1 --chi 47.8460 --time 500 --save_static_plots no --output_dir section_6-4 --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 47.9000 --time 1000 --save_static_plots no --output_dir section_6-4 --save_data no

#Subcritical case
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 95.7600 --time 30 --save_static_plots no --output_dir section_6-4 --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 3 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 95.7660 --time 30 --save_static_plots no --output_dir section_6-4 --save_data no
#

#Theorem 1.3: Chi increases
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 50.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 60.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 70.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 100.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 200.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 500.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
# ./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 1000.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no
./simulation.py --a 1 --b 1 --alpha 1 --m 1 --beta 2 --gamma 1 --mu 1 --nu 1 --L 1 --epsilon 0.01 --meshsize_abs 30 --eigen_mode_n 1 --chi 2000.0000 --time 1000 --save_static_plots no --output_dir chi_increases --save_data no

# # Second row table 3.2 T=100
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 7.50 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 7.80 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 7.95 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 8.30 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
#
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 11.00 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 11.50 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 11.70 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 12.00 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
#
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.00 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.40 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.70 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 19.00 --time 100 --save_static_plots no --output_dir table_3_2_2nd_row --save_data no
#
# # Third row table 3.2 T=100
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 15.00 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 15.50 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 16.50 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 17.00 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
#
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 16.00 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 16.50 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 17.50 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.00 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
#
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 38.00 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 38.40 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 39.20 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 39.70 --time 100 --save_static_plots no --output_dir table_3_2_3rd_row --save_data no

# # Second row table 3.2 T=30
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 7.50 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 7.80 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 7.95 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 1 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 8.30 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
#
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 11.00 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 11.50 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 11.70 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 5 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 12.00 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
#
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.00 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.40 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.70 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
# ./simulation.py --a 1 --b 10 --alpha 3 --m 1 --beta 1 --gamma 2 --mu 100 --nu 10 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 19.00 --time 30 --save_static_plots no --output_dir table_3_2_2nd_row_T30 --save_data no
#
# # Third row table 3.2 T=30
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 15.00 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 15.50 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 16.50 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 0 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 17.00 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
#
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 16.00 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 16.50 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 17.50 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 1 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 18.00 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
#
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 38.00 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 38.40 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 39.20 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
# ./simulation.py --a 3 --b 1 --alpha 2 --m 1 --beta 15 --gamma 2 --mu 50 --nu 1 --L 1 --epsilon 0.5 --meshsize_abs 100 --eigen_mode_n 1 --chi 39.70 --time 30 --save_static_plots no --output_dir table_3_2_3rd_row_T30 --save_data no
