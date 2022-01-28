<h1 align='center'>Neural SPDEs</h1>
<h2 align='center'>Resolution-Invariant Learning of Continuous Spatiotemporal Dynamics</h2>

Stochastic partial differential equations (SPDEs) are the mathematical tool of choice for modelling spatiotemporal PDE-dynamics under the influence of randomness. Based on the notion of mild solution of an SPDE, we introduce a novel neural architecture to learn solution operators of PDEs with (possibly stochastic) forcing from partially observed data. The proposed Neural SPDE model provides an extension to two popular classes of physics-inspired architectures. On the one hand, it extends Neural CDEs and variants -- continuous-time analogues of RNNs -- in that it is capable of processing incoming sequential information arriving irregularly in time and observed at arbitrary spatial resolutions. On the other hand, it extends Neural Operators -- generalizations of neural networks to model mappings between spaces of functions -- in that it can parameterize solution operators of SPDEs depending simultaneously on the initial condition and a realization of the driving noise. By performing operations in the spectral domain, we show how a Neural SPDE can be evaluated in two ways, either by calling an ODE solver (emulating a spectral Galerkin scheme), or by solving a fixed point problem. Experiments on various semilinear SPDEs, including the stochastic Navier-Stokes equations, demonstrate how the Neural SPDE model is capable of learning complex spatiotemporal dynamics in a resolution-invariant way, with better accuracy and lighter training data requirements compared to alternative models, and up to 3 orders of magnitude faster than traditional solvers.

---

## Structure of the repository

- `data` folder: contains notebooks to generate various datasets (stochastic Ginzburg Landau, KdV, Navier-Stokes) using numerical solvers for SPDEs (finite difference and spectral Galerkin methods)
- `torchspde` folder: contains the implementation of the Neural SPDE model
- `baselines` folder: contains the implementation of various models (NCDE, NRDE, FNO and DeepONet) to benchmark the NSPDE model 
- `examples` folder: contains notebooks to train and evaluate an NSPDE (and baselines models) on different SPDEs, and benchmark the NSPDE model

