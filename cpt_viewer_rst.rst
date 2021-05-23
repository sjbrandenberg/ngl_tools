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
Tables queried in this notebook, and the fields within those tables are described in the tables below.

List of Tables
==============

===== ===========
Table Description
===== ===========
SITE  Highest level table that serves as the organizational structure for project team collaboration
TEST  Table containing descriptions of tests including CPT, boreholes, geophysical tests, and groundwater measurements
SCPG  Table containing metadata about CPT test
SCPT  Table containing CPT test data
===== ===========

SITE Table
==========

========= ===========
Field     Description
========= ===========
SITE_ID   Primary key for the site table
SITE_NAME Site name (appears in site_widget dropdown)
========= ===========

TEST Table
==========

========= ===========
Field     Description
========= ===========
TEST_ID   Primary key for TEST table
SITE_ID   Foreign key from SITE table associating a test with a site
TEST_NAME Test name (appears in test_widget dropdown)
========= ===========

SCPG Table
==========

========= ===========
Field     Description
========= ===========
SCPG_ID   Primary key for SCPG table
TEST_ID   Foreign key from TEST table associating a cone penetration test with a test
SCPG_CSA  Surface area of the cone tip in square centimeters
SCPG_RATE Nominal rate of penetration of the cone in cm/s
SCPG_CREW Name of logger / organization
SCPG_METH Penetration method
SCPG_STAR Start date of activity
SCPG_ENDD End date of activity
SCPG_PWP  Position of pore pressure measurement on cone
SCPG_REM  Remarks
========= ===========

SCPT Table
==========

========= ===========
Field     Description
========= ===========
SCPT_ID   Primary key for SCPT Table
SCPG_ID   Foreign key from SCPG table associating cone penetratin test data with test metadata
SCPG_DPTH Depth of CPT measurement in m
SCPT_RES  Cone tip resistance (qc) in MPa
SCPT_FRES Sleeve friction resistance (fs) in MPa
SCPT_PWP  Pore-water pressure in MPa
========= ===========

----
Code
----

Import packages
===============

In this case, we need to import ipywidgets, matplotlib, numpy, ngl_db, and pandas. The "%matplotlib notebook" magic renders an interactive plot in the notebook.

.. code-block:: python

   %matplotlib notebook
   import ipywidgets as widgets
   from matplotlib import pyplot as plt
   import numpy as np
   import ngl_db
   import pandas as pd

Connect to database
===================

.. code-block:: python
   
    cnx = ngl_db.connect()
    
Query distinct SITE_ID and SITE_NAME for sites that have CPT data
=================================================================
The query below finds distinct SITE_ID and SITE_NAME fields that contain CPT data for the purpose of populating the site dropdown widget. 
INNER JOIN commands are required between SITE, TEST, and SCPG to find sites containing CPT data.
A site might contain more than one CPT test, but we do not want replicated fields in the site dropdown widget. Therefore we use the "DISTINCT" command.

.. code-block:: python

    sql = 'SELECT DISTINCT SITE.SITE_ID, SITE.SITE_NAME FROM SITE INNER JOIN TEST ON SITE.SITE_ID = TEST.SITE_ID INNER JOIN SCPG ON SCPG.TEST_ID = TEST.TEST_ID'
    site_df = pd.read_sql_query(sql, cnx)
    
