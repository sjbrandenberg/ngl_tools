
Oct 2021 DesignSafe Webinar
============================

The DesignSafe_Webinar_Oct2021 notebook was created during a webinar/workshop hosted by DesignSafe and the Pacific Earthquake Engineering Research (PEER) center.

The notebook demonstrates the following:

1. Connecting to NGL database in DesignSafe
2. Querying data from SITE, TEST, SCPG, and SCPT tables into Pandas dataframes
3. Plotting data from the selected test using matplotlib

Cone penetration test data plotted in the notebook include tip resistance, sleeve friction, and pore pressure. In some cases, sleeve friction and pore pressure are not measured, in which case the plots are empty.


Related Links
----------------

[DesignSafe Webinar Recording](https://youtu.be/TNOPOU4lx5w)

[DesignSafe Workshop Recording](https://youtu.be/_nKpSqa1rso)


Jupyter notebook
----------------

[Jupyter notebook on DesignSafe](https://jupyter.designsafe-ci.org/user/name/tree/CommunityData/NGL/DesignSafe_Webinar_Oct2021.ipynb)


Tables
------

Tables queried in this notebook, and the fields within those tables are described in the tables below.


### List of Tables

|Table | Description|
|--- |---|
|SITE | Highest level table that serves as the organizational structure for project team collaboration |
|TEST | Table containing descriptions of tests including CPT, boreholes, geophysical tests, and groundwater measurements |
|SCPG | Table containing metadata about cone penetrometer test (CPT) |
|SCPT | Table containing CPT data |
|WATR | Table containing ground water table information |


### SITE Table

|Field  | Description|
| --- | --- |
|SITE_ID   |Primary key for the SITE table|
|SITE_NAME |Site name|


### TEST Table

|Field | Description|
| --- | --- |
|TEST_ID|   Primary key for TEST table|
|SITE_ID|   Foreign key from SITE table associating a test with a site|
|TEST_NAME| Test name|


### SCPG Table

|Field |    Description|
| --- | --- |
|SCPG_ID|   Primary key for SCPG table|
|TEST_ID |  Foreign key from TEST table associating the SCPG with a TEST|


### SCPT Table

|Field |    Description|
| --- | --- |
|SCPT_ID   |Primary key for SCPT Table|
|SCPG_ID   |Foreign key from SCPG table associating SCPT with SCPG metadata|
|SCPT_DPTH |Depth of measurement in m|
|SCPT_RES  |Cone tip resistance (qc) in MPa|
|SCPT_FRES |Sleeve friction resistance (fs) in MPa|
|SCPT_PWP  |Porewater pressure in MPa|


### WATR Table

|Field|     Description|
| --- | --- |
|WATR_ID   |Primary key for WATR Table |
|TEST_ID   |Foreign key from TEST table associating WATR with TEST metadata |
|WATR_DPTH |Depth of measurement in m |


Code
----

This section describes the [Jupyter notebook](https://jupyter.designsafe-ci.org/user/name/notebooks/CommunityData/NGL/DesignSafe_Webinar_Oct2021.ipynb) available via DesignSafe. The code is broken into chunks with explanations of each section of code.


### Connect to NGL Database
1) import the ngl_db package and 
2) create a connection object to ngl_db called cnx


```python
import ngl_db

cnx = ngl_db.connect()
```

### Query SITE Table Using Pandas
An easy way to query the database is to use the Pandas read_sql command, which queries data and returns a Pandas dataframe.

1) import the Pandas package, 
2) create a string called sql that queries all information from the SITE table, and 
3) create a Pandas dataframe called df that contains the results of the query.


```python
import pandas as pd

sql = "SELECT * FROM SITE"
df = pd.read_sql(sql,cnx)
df
```


### Query all TESTs for a given SITE
This cell queries the TEST table looking for all TESTs with the same SITE_ID


```python
site_id = 159
sql = 'SELECT * FROM TEST where TEST.SITE_ID = "{}"'.format(site_id)
TESTdf = pd.read_sql(sql,cnx)
TESTdf
```


### Query CPT Metadata (SCPG) for a given TEST
This cell queries the SCPG table for a single CPT test


```python
test_id = TESTdf['TEST_ID'][1]
sql = 'SELECT * FROM SCPG where SCPG.TEST_ID = "{}"'.format(test_id)
SCPGdf = pd.read_sql(sql,cnx)
SCPGdf
```


### Plot CPT Data (SCPT) for a given TEST
This cell uses matplotlib to plot CPT data located in the SCPT table


```python
%matplotlib notebook
import matplotlib.pyplot as plt

#get CPT data for a given SCPG_ID, and load into Pandas dataframe
scpg_id = SCPGdf['SCPG_ID'][0]
sql = 'SELECT * FROM SCPT where SCPT.SCPG_ID = "{}"'.format(scpg_id)
SCPTdf = pd.read_sql(sql,cnx)

#plot cone tip resistance, friction resistance, and pore pressures
fig,axs = plt.subplots(ncols=3, figsize=(7,6),sharey=True)
axs[0].invert_yaxis() #moves zero depth to the top of the plot
axs[0].plot(SCPTdf['SCPT_RES'],SCPTdf['SCPT_DPTH'])
axs[1].plot(SCPTdf['SCPT_FRES'],SCPTdf['SCPT_DPTH'])
axs[2].plot(SCPTdf['SCPT_PWP'],SCPTdf['SCPT_DPTH'])
axs[0].set_xlabel('Cone Tip Resistance (MPa)')
axs[1].set_xlabel('Sleeve Friction (MPa)')
axs[2].set_xlabel('Pore Pressure (MPa)')
axs[0].set_ylabel('Depth (m)')
for ax in axs:
    ax.grid(True, alpha=0.5)
plt.tight_layout()
```


### Get WATR information for given TEST_ID
This cell extracts the depth to groundwater from the WATR table for the same TEST_ID specified earlier


```python
sql = 'SELECT * FROM WATR'
sql += ' Where WATR.TEST_ID = "{}"'.format(test_id)
waterdf = pd.read_sql(sql,cnx)
z_gwt = waterdf['WATR_DPTH'].values[0]
waterdf
```


### Put it all together!
This cell puts everything together in one cell, and adds horizontal lines representing the groundwater table to the plot.


```python
import ngl_db
import pandas as pd
import matplotlib.pyplot as plt

cnx = ngl_db.connect()

#Get list of TESTs for given SITE_ID
site_id = 159
sql = 'SELECT * FROM TEST where TEST.SITE_ID = "{}"'.format(site_id)
TESTdf = pd.read_sql(sql,cnx)

#Get SCPG_ID for given TEST_ID
test_id = TESTdf['TEST_ID'][1]
sql = 'SELECT * FROM SCPG where SCPG.TEST_ID = "{}"'.format(test_id)
SCPGdf = pd.read_sql(sql,cnx)

#get SCPT data for a given SCPG_ID, and load into Pandas dataframe
scpg_id = SCPGdf['SCPG_ID'][0]
sql = 'SELECT * FROM SCPT where SCPT.SCPG_ID = "{}"'.format(scpg_id)
SCPTdf = pd.read_sql(sql,cnx)

#get WATR data for same TEST_ID
sql = 'SELECT * FROM WATR'
sql += ' Where WATR.TEST_ID = "{}"'.format(test_id)
waterdf = pd.read_sql(sql,cnx)
z_gwt = waterdf['WATR_DPTH'].values[0]

#plot cone tip resistance, friction resistance, and pore pressures, with horizontal line for GWT
fig,axs = plt.subplots(ncols=3, figsize=(7,6),sharey=True)
axs[0].invert_yaxis() #moves zero depth to the top of the plot
axs[0].plot(SCPTdf['SCPT_RES'],SCPTdf['SCPT_DPTH'])
axs[1].plot(SCPTdf['SCPT_FRES'],SCPTdf['SCPT_DPTH'])
axs[2].plot(SCPTdf['SCPT_PWP'],SCPTdf['SCPT_DPTH'])
axs[0].set_xlabel('Cone Tip Resistance (MPa)')
axs[1].set_xlabel('Sleeve Friction (MPa)')
axs[2].set_xlabel('Pore Pressure (MPa)')
axs[0].set_ylabel('Depth (m)')
for ax in axs:
    ax.grid(alpha=0.5)
    ax.axhline(z_gwt,color='b')
plt.tight_layout()
```


### Query all SITE and TEST fields that have both SCPG and WATR
If you want to find another SITE_ID/TEST_ID/SCPG_ID combination to try with this notebook, you can use a JOIN statement to combine the SITE, TEST, SCPG, and WATR tables to find tests where there is CPT information and groundwater table information


```python
sql = 'SELECT SITE.SITE_ID, SITE.SITE_NAME, TEST.TEST_ID, TEST.TEST_NAME, SCPG.SCPG_ID, WATR.WATR_ID '
sql += 'FROM SITE INNER JOIN TEST ON TEST.SITE_ID = SITE.SITE_ID '
sql += 'INNER JOIN SCPG ON SCPG.TEST_ID = TEST.TEST_ID '
sql += 'INNER JOIN WATR ON WATR.TEST_ID = TEST.TEST_ID'

test_metadata = pd.read_sql(sql, cnx)
test_metadata
```


### Close the connection
Close the connection to the NGL database when you're done with your queries


```python
cnx.close()
```
