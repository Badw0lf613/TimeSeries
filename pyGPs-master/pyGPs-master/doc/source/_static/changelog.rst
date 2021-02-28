Changelog pyGPs v1.3.1
==========================

December 15th 2014
----------------------------

- pyGPs added to pip
- mathematical definitions of kernel functions available in documentation
- more error message added


November 25th 2014
----------------------------

structural updates:

- full inline documentation with input parameter and output specified
- check for the inputs and provide diagnostic messages for some inputs
- consistant naming in inline and online documentation
- string representation for dnlZStruct and postStruct 
  - Now you can do sth like,
  - nlZ, dnlZ, post = model.getPosterior(x,y)
  - print post
  - instead of a python object, we provide now a more informative description.
- add optimization into unit test routines. Also add checking for cholesky decomposition and checking positive-definite property of kernel matrix.
- add jitter to the digonal of linear, linARD, and poly covariance for numerical stability.
- fix several minor problems in unit test framework
- hierachically rearranged for online documentation
- add several supplementary instruction in online documentation


October 19th 2014
----------------------


documentation updates:

- DOC: model.fit() is now named model.getPosterior
- DOC: typo fixed: cov.LIN changed to cov.Linear
- DOC: removed cov.Periodic() in demos because its limited in 1-d data.
- API file updated accordingly


structural updates:

- removed unused ScalePrior attribute in most inference method
- added function jitchol, which added a small jitter(instead of doing Cholesky decomposition directly) to the diagonal when the kernel matrix is ill conditioned.
- thrown error when using periodic covariance functions for non-1d data. We also removed cov.Periodic() in demos because its limited usage.
- combined equally spaced positions of inputs as test positions as well in plot methods to get a more accurate plotting.
- rename model.fit() to model.getPosterior(), while model.optimize() stays the same. (since it is confusing for some users that the name fit() is not doing optimizing.)


August 9th 2014
------------------


structural updates:

- added SM covariance and Gabor covariance
- bug fixed in dfunc() in Matern covariance
- change proceed() to evaluate() for inference and likelihood classes
- added unit test module for SM covariance and Gabor covariance





Changelog pyGPs v1.2
=======================


July 14th 2014
------------------

documentation updates:

- online docs updated
- API file updated

structural updates:

- made private for methods that users don't need to call



July 8th 2014
----------------

structural updates:

- add hyperparameter(signal variance s2) for linear covariance
- add unit testing for inference,likelihood functions as well as models
- NOT show(print) "maximum number of sweep warning in inference EP" any more
- documentation updated

bug fixes:

- typos in lik.Laplace
- derivative in lik.Laplace




June 30th 2014
----------------

structural updates:

- input target now can either be in 2-d array with size (n,1) or in 1-d array with size (n,)
- setup.py updated
- "import pyGPs" instead of "from pyGPs.Core import gp"
- rename ".train()" to ".optimize()"
- rename "Graph-stuff" to "graphExtension"
- rename kernelOnGraph to "nodeKernels" and graphKernel to "graphKernels"
- redundancy removed for model.setData(x,y)
- rewrite "mean.proceed()" to "getMean()" and "getDerMatrix()"
- rewrite "cov.proceed()" to "getCovMatrix()" and "getDerMatrix()"
- rename cov.LIN to cov.Linear (to be consistent with mean.Linear)
- rename module "valid" to "validation"
- add graph dataset Mutag in python file. (.npz and .mat)
- add graphUtil.nomalizeKernel()
- fix number of iteration problem in graphKernels "PropagationKernel"
- add unit testing for covariance, mean functions



bug fixes:

- derivatives for cov.LINard
- derivative of the scalar for cov.covScale
- demo_GPR_FITC.py missing pyGPs.mean






