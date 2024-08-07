This repository provides code that can be used for a basic simulation of MRF signals as well as the optimization of MRF flip angles and repetition times. It has mainly been created for my masters thesis ("Target-dependent Optimization of Magnetic Resonance Fingerprinting Sequences", Universitaet Wuerzburg, 2023). Further details can also be found in my 2024 ISMRM Abstract #3547, "Towards Sequence Optimization for Multi-Compartment Magnetic Resonance Fingerprinting".

Tom Griesler, 05/24

tomgr@umich.edu


OVERVIEW
- signalmodel_bloch.py: code for MRF signal simulation and cost function calculation using the isochromat summation approach and Bloch equations.
- signalmodel_epg.py: code for MRF signal simulation and cost function calculation in an EPG implementation.
- slsqp.py: implementation of the SLSQP algorithm for MRF sequence optimization. 
- tools.py: contains a helper function that is used in multiple scripts. 
- initialization: example flip angles and repetition times.
- demo.ipynb: jupyter notebook that shows how to use the code to simulate and optimize MRF sequences.
- iso_vs_epg.ipynb: jupyter notebook that shows that the isochromat summation and EPG implementations lead to equivalent results. 
- environment.yml: environment that has been used to test this code. Basically, you will need to install numpy, torch, and scipy to run the code, matplotlib for visualization and ipykernel if you want to run code in the IPython kernel for Jupyter. The rest are dependencies.


GENERAL REMARKS

It took me quite a while to get all the details of the Pytorch and SLSQP optimization implementation working, so I hope to save you some time by providing this code. I tried to make it easily understandable by creating docstrings and comments. If you have any further questions or issues, please contact me by email and I'll be happy to help as best as I can. 

Please be aware that while I tried to check the code for implementation errors, I have not put a ton of time into optimizing the code for efficiency. This was one of my first big coding projects, and while I tried to follow PEP 8 recommendations, I'm sure there are more pythonic ways to implement things. 


FUTURE WORK   
- I plan to add some checks to throw an error when functions are called with wrong/inconsistent input parameters.
- if you miss a feature or don't like a certain implementation, please let me know and I'll see what I can do. 


REFERENCES

The implementation of the isochromat summation signal model and single compartment CRLB calculation is based on the following paper:
    Bo Zhao, Haldar JP, Congyu Liao, Dan Ma, Yun Jiang, Griswold MA, Setsompop K, Wald LL. Optimal Experiment Design for Magnetic Resonance Fingerprinting: Cramér-Rao Bound Meets Spin Dynamics. IEEE Trans Med Imaging. 2019 Mar;38(3):844-861

Based on this, I added an EPG-based implementation which leads to a significant reduction of computation times (while yielding equivalent results). I also added two other cost function which are derived from the two following publications: 
    - Cohen O, Rosen MS. Algorithm comparison for schedule optimization in MR fingerprinting. Magn Reson Imaging. 2017 Sep;41:15-21
    - Heesterbeek D, Vos F, van Gijzen M, Nagtegaal M. Sequence Optimisation for Multi-Compartment Analysis in Magnetic Resonance Fingerprinting. ISMRM 2021; Abstract #1561

The implementation of the SLSQP optimization is based in part on code that has been published with the paper:
    Lee PK, Watkins LE, Anderson TI, Buonincontri G, Hargreaves BA. Flexible and efficient optimization of quantitative sequences using automatic differentiation of Bloch simulations. Magn Reson Med. 2019 Oct;82(4):1438-1451

The example flip angles and repetition times have been published with the paper:
    Cao X, Liao C, Iyer SS, Wang Z, Zhou Z, Dai E, Liberman G, Dong Z, Gong T, He H, Zhong J, Bilgic B, Setsompop K. Optimized multi-axis spiral projection MR fingerprinting with subspace reconstruction for rapid whole-brain high-isotropic-resolution quantitative imaging. Magn Reson Med. 2022 Jul;88(1):133-150
