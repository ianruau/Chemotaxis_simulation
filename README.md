# Chemotaxis Simulations

This repository contains a program that simulates a chemotaxis process from a parabolic-elliptic system of partial differential equations. It is based on the research paper *Global Existence and Asymptotic Behaviour of Classical Solutions of Chemotaxis Models with Signal-dependent Sensitivity and Logistic Source* by Dr. Wenxian Shen, Dr. Le Chen, and their PhD student Ian Ruau. Here is a link to the research paper **[link to paper]**

Simulations for special sets of parameters are consistent with proven theoretical results. The advantage of this package is that it helps us get insights about behaviours outside of the sets of parameters considered in the research paper.

<!-- ## Features -->
<!---->
<!-- - Comprehensive simulations of different surface growth models. -->
<!-- - Easy-to-use interface for conducting and analyzing simulations. -->
<!-- - Detailed documentation for understanding and extending the simulations. -->

## Installation

To get started with these simulations, you can install the package using pip:

```bash
pip install NAME-OF-THE-PACKAGE
```

## Sample Usage

Once the package is installed, the user can access the help prompt by simply
running the following command

```bash
./simulation.py --help
```

Here is an example of a simulation

```bash
./simulation.py --chi 30 --meshsize 100 --time 5 --eigen_index 2 --epsilon 0.5 --generate_video yes
```

The command above will show first some characteristic constants of the system of
partial differential equations that depend on the parameters input by the user:

![constants](./homepage/chi_table.png)

A terminal plot of the initial functions u(0,x) and v(0,x) is also shown on the
terminal:

![initial_plots](./homepage/u_v_terminal_plots.png)

Once the simulation is complete, a picture of the results is saved in both .png
and .jpeg formats:

![images_saved](./homepage/images_saved.png)

![Plots](./homepage/a=1-0_b=1-0_alpha=1-0_m=1-0_beta=1-0_chi=30-0_mu=1-0_nu=1_gamma=1-0_meshsize=100_time=5-0_epsilon=0-5_eigen_index=2.png)

If the user chose to save the animation of the chemotaxis process, an .mp4 video
will be saved as well:

![Video](./homepage/video.gif)

## Documentation

For detailed information about the package and its functionalities, visit **Link
to documentation page**

<!-- ## How to Contribute -->
<!---->
<!-- Contributions to this project are welcome! To contribute, please: -->
<!---->
<!-- 1. Fork the repository. -->
<!-- 2. Create a new branch for your feature. -->
<!-- 3. Add your changes and commit them. -->
<!-- 4. Push to the branch. -->
<!-- 5. Create a new pull request. -->

## References

**Add references**

## License

This project is licensed under the **Insert license** file for details.

## Contact

For any queries or further discussion, feel free to contact us at

- Le Chen: [chenle02@gmail.com] or [le.chen@auburn.edu].
- Ian Ruau: [ian.ruau@auburn.edu].
