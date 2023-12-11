#########################
 PyGraphon Documentation
#########################

Welcome to the documentation of `PyGraphon
<https://github.com/dufourc1/pygraphon>`_ for estimating graphons from
observed data! Our library provides a powerful set of tools for
estimating graphons, which are a type of object used to model large,
random graphs, from observed data.

.. code:: python

   from pygraphon import HistogramEstimator

   #  sample a graph with n nodes from one of the built-in graphons f(x,y) = x*y
   from pygraphon import graphon_graphon_product

   A = graphon_product.draw(n=100)

   # Create a histogram estimator
   estimator = HistogramEstimator()

   # Fit the estimator to a graph with adjacency matrix A
   estimator.fit(A)

   # Get the estimated graphon
   graphon_estimated = estimator.get_graphon()

   # get the estimated block connectivity matrix
   theta = graphon_estimated.get_theta()

   # get the estimated edge probability matrix
   P_estimated = graphon_estimated.get_edge_connectivity()

To cite this library, please use the following:

   Dufour, C., & Verdeyme, A. (2023). PyGraphon (v0.4). Zenodo.
   https://doi.org/10.5281/zenodo.10355248

.. code:: latex

   @misc{dufour_2023,
        title = {{{PyGraphon}}},
        author = {Dufour, Charles and Verdeyme, Arthur},
        year = {2023},
        month = dec,
        doi = {10.5281/zenodo.10355248},
        howpublished = {Zenodo}
    }


.. toctree::
   :maxdepth: 1

   installation
   tutorials
   api
   references

********************
 Indices and Tables
********************

-  :ref:`genindex`
-  :ref:`modindex`
-  :ref:`search`
