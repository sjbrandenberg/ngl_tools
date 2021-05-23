============================
Cone penetration test viewer
============================

The cone penetration test viewer demonstrates the following:

1. Connecting to NGL database in DesignSafe
2. Querying data from SITE, TEST, SCPG, and SCPT tables into Pandas dataframes
3. Creating dropdown widgets using the ipywidgets package to allow users to select site and test data
4. Using the ipywidgets "observe" feature to call functions when users select a widget value
5. Plotting data from the selected cone penetration test using matplotlib

Cone penetration test data plotted in the notebook include tip resistance, sleeve friction, and pore pressure. In some cases, sleeve friction and pore pressure are not measured, in which case the plots are empty.

----------------
Jupyter notebook
----------------
`Jupyter notebook on DesignSafe <https://jupyter.designsafe-ci.org/user/sjbrande/notebooks/CommunityData//NGL/CPT_viewer.ipynb>`_

------
Tables
------
Tables queried in this notebook are described in the table below.

===== ===========
Table Description
===== ===========
SITE  Highest level table that serves as the organizational structure for project team collaboration
TEST  Table containing descriptions of tests including CPT, boreholes, geophysical tests, and groundwater measurements
SCPG  Table containing metadata about CPT test
SCPT  Table containing CPT test data
===== ===========
