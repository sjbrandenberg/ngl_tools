ngl_tools documentation
=======================

ngl_tools is a collection of Jupyter notebooks developed to interact with the NGL database in DesignSafe. 
The Next Generation Liquefaction (NGL) Project is advancing the state of the art in liquefaction research 
and working toward providing end users with a consensus approach to assess liquefaction potential within 
a probabilistic and risk-informed framework. Specifically, NGL’s goal is to first collect and organize 
liquefaction information in a common and comprehensive database to provide all researchers with a 
substantially larger, more consistent, and more reliable source of liquefaction data than existed previously. 
Based on this database, we will create probabilistic models that provide hazard- and risk-consistent bases 
for assessing liquefaction susceptibility, the potential for liquefaction to be triggered in susceptible soils, 
and the likely consequences. NGL is committed to an open and objective evaluation and integration of data, 
models and methods, as recommended in a 2016 National Academies `report <https://www.nap.edu/catalog/23474/state-of-the-art-and-practice-in-the-assessment-of-earthquake-induced-soil-liquefaction-and-its-consequences>`_. 
The evaluation and integration of the data into new models and methods will be clear and transparent. Following these principles will ensure 
that the resulting liquefaction susceptibility, triggering, and consequence models are reliable, robust and 
vetted by the scientific community, providing a solid foundation for designing, constructing and overseeing 
critical infrastructure projects.

The NGL database is populated through a web GUI at www.nextgenerationliquefaction.org/. The web interface 
provides limited capabilities for users to interact with data. Users are able to view and download data, 
but they cannot use the GUI to develop an end-to-end workflow to make scientific inferences and draw conclusions 
from the data. To facilitate end-to-end workflows, the NGL database is replicated daily to `DesignSafe <https://www.designsafe-ci.org>`_, where 
users can interact with it using Jupyter notebooks.

External Links
______________
`https://www.designsafe-ci.org <https://www.designsafe-ci.org>`_

`https://www.nextgenerationliquefaction.org <https://www.nextgenerationliquefaction.org>`_


.. toctree::
   :caption: Background Info
   :maxdepth: 1
   

   connect_to_database
   published_notebooks
   understanding_schema
   
.. toctree::
   :caption: Jupyter Notebooks
   :maxdepth: 1
   
   example_queries
   cpt_viewer
   Vs_invasive_viewer
