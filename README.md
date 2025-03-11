# OCP(Open Cryptanalysis Platform)

The OCP tool is composed of the following main modules:

- **Tool** is the main file of the tool, offering interfaces to other modules. Users can launch here scripts to analyze primitives. 

- **operators** contains the various operators that can be used to build the modelisation of a `primitive`. Users can add here new operators if needed, or add new ways for the operators to be modeled. 
   
- **variables** contains the variable class that can be used to build the modelisation of a `primitive`.

- **primitives** contains the modelisation of various primitives, including permutations and block ciphers. Users can add here own primitive if not already present.

- **attack** contains various pre-defined attacks, including differential attacks. Users can perform attacks run on pre-difined primitives.

- **solving** contains automated solving techniques, including MILP, SAT, and CP solvers.

- **Implementations** contains the implementations of various primitives.

- **visualizations** contains the methods to display primitives and distinguishers.


The overall structure is illustrated in the diagram below:
<p align="center">
  <img src="docs/images/architecture.png" alt="architecture" width="600">
</p>



📖 For detailed documentation, visit the [OCP Wiki](https://github.com/Open-CP/OCP/wiki).  




