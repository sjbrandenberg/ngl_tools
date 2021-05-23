===============
Example Queries
===============

This document demonstrates how to query the NGL database using DesignSafe. The queries begin simple and become progressively more complicated.

-----------------------------
Query site table using pandas
-----------------------------

An easy way to query the database is to use the Pandas read_sql command, which queries data and returns a Pandas dataframe. 
The commands below imports the Pandas and ngl_db packages, creates a connection object to ngl_db called cnx, creates a 
string called sql that queries all information from the SITE table, and creates a Pandas dataframe called df that contains 
the results of the query.

.. code-block:: python

  import pandas as pd
  import ngl_db

  cnx = ngl_db.connect()
  sql = "SELECT * FROM SITE"
  df = pd.read_sql(sql,cnx)
  df

The output from the command is illustrated in the figure below. When this query was written, there were a total of 333 sites in 
the NGL database. The SITE_ID field is not contiguous because sites are sometimes deleted from the database, and the 
SITE_ID field is never re-used. The Pandas dataframe is broken between SITE_ID 151 and 677 for ease of displaying 
information in the output window. Many rows of data are not displayed in Figure 2 as a result.

.. figure:: images/SiteTableQuery.png
  :alt: Screenshot of result of query of SITE table data.
  
  **Figure 2.** Results of query of SITE table data.

--------------------------------------
Query Wildlife liquefaction array data
--------------------------------------

This cell queries event information from the EVNT table and surface evidence of liquefaction information from the FLDM table at the Wildlife Array site. The definition of each table and site is below. The query utilizes an INNER JOIN statement to combine tables based on shared keys, and will return all values that have matching keys in both tables. For more details, see `https://www.w3schools.com/sql/sql_join_inner.asp <https://www.w3schools.com/sql/sql_join_inner.asp>`_  

===== ===========
Table	Description
===== ===========
EVNT	Earthquake event information
FLDM	Field evidence of liquefaction information at a point within a site
FLDO	Field evidence of liquefaction information at a site
SITE	A site is the highest level organizational structure for information in the database
===== ===========

========= ===========
Field	    Description
========= ===========
EVNT_MAG	Earthquake Magnitude
EVNT_NM	  Event Name
EVNT_YR	  Event Year
FLDM_LAT	Latitude of manifestation observation
FLDM_LON	Longitude of manifestation observation
FLDM_SFEV	Indication of whether surface manifestation occurred (0 = no, 1 = yes)
FLDM_DESC	Description of liquefaction manifestation
========= ===========

.. code-block:: python
  
  import pandas as pd
  import ngl_db

  cnx = ngl_db.connect()

  sql = 'SELECT EVNT.EVNT_MAG, EVNT.EVNT_NM, EVNT.EVNT_YR, FLDM.FLDM_LAT, FLDM.FLDM_LON, FLDM.FLDM_SFEV, FLDM.FLDM_DESC '
  sql += 'FROM FLDO INNER JOIN FLDM on FLDO.FLDO_ID = FLDM.FLDO_ID '
  sql += 'INNER JOIN EVNT ON EVNT.EVNT_ID = FLDO.EVNT_ID '
  sql += 'INNER JOIN SITE ON FLDO.SITE_ID = SITE.SITE_ID '
  sql += 'WHERE SITE_NAME = "Wildlife Array"'

  df = pd.read_sql_query(sql, cnx)
  pd.set_option('display.max_colwidth', 100)
  df

.. figure:: images/WildlifeQuery1.png
  :alt: Screenshot of result of query of Wildlife liquefaction array query of event information and field observations.
  
  **Figure 3.** Screenshot of result of query of Wildlife liquefaction array query of event information and field observations.
