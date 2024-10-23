# Cone penetration test viewer

The cone penetration test viewer demonstrates the following:

1. Connecting to NGL database in DesignSafe
2. Querying data from SITE, TEST, SCPG, and SCPT tables into Pandas dataframes
3. Creating dropdown widgets using the ipywidgets package to allow users to select site and test data
4. Creating HTML widget for displaying metadata after a user select a test
5. Using the ipywidgets "observe" feature to call functions when users select a widget value
6. Plotting data from the selected cone penetration test using matplotlib

Cone penetration test data plotted in the notebook include tip resistance, sleeve friction, and pore pressure. In some cases, sleeve friction and pore pressure are not measured, in which case the plots are empty.

## Jupyter notebook
[Jupyter notebook on DesignSafe](https://jupyter.designsafe-ci.org/user/name/tree/CommunityData/NGL/CPT_viewer.ipynb)

## Tables
Tables queried in this notebook, and the fields within those tables are described in the tables below.

### List of Tables

| Table | Description |
| ----- | ----------- |
| SITE  | Highest level table that serves as the organizational structure for project team collaboration |
| TEST  | Table containing descriptions of tests including CPT, boreholes, geophysical tests, and groundwater measurements |
| SCPG  | Table containing metadata about CPT test |
| SCPT  | Table containing CPT test data |

### SITE Table

| Field   | Description |
| -----   | ----------- |
| SITE_ID | Primary key for the site table |
| SITE_NAME | Site name (appears in site_widget dropdown) |

### TEST Table

| Field   | Description |
| -----   | ----------- |
| TEST_ID |  Primary key for TEST table |
| SITE_ID |  Foreign key from SITE table associating a test with a site |
| TEST_NAME | Test name (appears in test_widget dropdown) |

### SCPG Table

| Field   | Description |
| -----   | ----------- |
| SCPG_ID | Primary key for SCPG table |
| TEST_ID |  Foreign key from TEST table associating a cone penetration test with a test |
| SCPG_CSA | Surface area of the cone tip in square centimeters |
| SCPG_RATE | Nominal rate of penetration of the cone in cm/s |
| SCPG_CREW | Name of logger / organization |
| SCPG_METH | Penetration method |
| SCPG_STAR | Start date of activity |
| SCPG_ENDD | End date of activity |
| SCPG_PWP  | Position of pore pressure measurement on cone |
| SCPG_REM  | Remarks |

### SCPT Table

| Field   | Description |
| -----   | ----------- |
| SCPT_ID |   Primary key for SCPT Table   |
| SCPG_ID |   Foreign key from SCPG table associating cone penetratin test data with test metadata | 
| SCPG_DPTH | Depth of CPT measurement in m |
| SCPT_RES  | Cone tip resistance (qc) in MPa |
| SCPT_FRES | Sleeve friction resistance (fs) in MPa | 
| SCPT_PWP  | Pore-water pressure in MPa |

## Code

This section describes the [Jupyter notebook](https://jupyter.designsafe-ci.org/user/name/notebooks/CommunityData/NGL/CPT_viewer.ipynb) available via DesignSafe. The code is broken into chunks with explanations of each section of code.

### Import packages

In this case, we need to import ipywidgets, matplotlib, numpy, ngl_db, and pandas. The "%matplotlib notebook" magic works in Jupyter notebooks and the "%matplotlib widget" magic works in Jupyter Lab, and renders an interactive plot in the notebook.

```python
try:
    %matplotlib widget
except:
    try:
        %matplotlib notebook
    except:
        print("Falling back to '%matplotlib inline'")

import pandas as pd
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import numpy as np
import designsafe_db.ngl_db as ngl
```

### Query distinct SITE_ID and SITE_NAME for sites that have CPT data
The query below finds distinct SITE_ID and SITE_NAME fields that contain CPT data for the purpose of populating the site dropdown widget. 
INNER JOIN commands are required between SITE, TEST, and SCPG to find sites containing CPT data.
A site might contain more than one CPT test, but we do not want replicated fields in the site dropdown widget. Therefore we use the "DISTINCT" command.

```python
sql = 'SELECT DISTINCT SITE.SITE_ID, SITE.SITE_NAME FROM SITE INNER JOIN TEST ON SITE.SITE_ID = TEST.SITE_ID INNER JOIN SCPG ON SCPG.TEST_ID = TEST.TEST_ID'
site_df = ngl.read_sql(sql)
```

### Create key, value pairs for SITE_NAME and SITE_ID, and create site_widget
Dropdown widgets accept key-value pairs for the "options" field. This is desireable here because the SITE_ID can be set to the key, and subsequently utilized in queries when a user selects a site. The code below converts queried site data into name, value pairs.

```python
site_df.set_index('SITE_ID',inplace=True)
site_df.sort_values(by='SITE_NAME',inplace=True)
site_options = [('Select a site', -1)]
for key, value in site_df['SITE_NAME'].to_dict().items():
    site_options.append((value, key))
site_widget = widgets.Dropdown(options=site_options, description='Site')
``` 

### Create empty test_widget. This widget will get populated when a site is selected

```python
test_options = [('Select a test', -1)]
test_widget = widgets.Dropdown(options=test_options, description='Test', disabled=True)
widget_box= widgets.VBox([site_widget, test_widget])
display(widget_box)
```

### Create plot objects and initialize empty plots

```python
plot_widget = widgets.Output()
with plot_widget:
    fig, ax = plt.subplots(1, 3, figsize=(6,4), sharey='row')
    line1, = ax[0].plot([], [])
    ax[0].set_xlabel('qc (MPa)')
    ax[0].set_ylabel('depth (m)')
    ax[0].grid(True)
    ax[0].invert_yaxis()
    line2, = ax[1].plot([], [])
    ax[1].set_xlabel('fs (MPa)')
    ax[1].grid(True)
    ax[1].invert_yaxis()
    line3, = ax[2].plot([], [])
    ax[2].set_xlabel('u2 (MPa)')
    ax[2].grid(True)
    ax[2].invert_yaxis()
    fig.tight_layout()
    plt.show(fig)
```

### Create empty metadata_widget. This widget will get populated when a CPT test is selected

```python
metadata_widget = widgets.HTML(value='')
```

### Define function for populating test_widget when a user selects a site from the site_widget dropdown

This code sets data for the plots to be empty, and sets the metadata widget to be empty as well. If the top-level field is selected (i.e., 'Select a Test'), then the test_widget is disabled.
If a site is selected, a SQL query is made on all of the CPT tests for that site, and the test dropdown is populated.

```python
def on_site_widget_change(change):
   ax[0].lines.clear()
    ax[1].lines.clear()
    ax[2].lines.clear()
    metadata_widget.value=''
    if(change['new']==-1):
        test_widget.options = [('Select a test', -1)]
        test_widget.disabled = True
    else:
        test_options = [('Select a test', -1)]
        sql = 'SELECT DISTINCT TEST.TEST_ID, TEST.TEST_NAME FROM TEST INNER JOIN SCPG ON TEST.TEST_ID = SCPG.TEST_ID WHERE TEST.SITE_ID = ' + str(change['new'])
        test_df = ngl.read_sql(sql)
        test_df.set_index('TEST_ID',inplace=True)
        test_df.sort_values(by='TEST_NAME',inplace=True)
        for key, value in test_df['TEST_NAME'].to_dict().items():
            test_options.append((value, key))
        test_widget.options = test_options
        test_widget.disabled = False
    return
```

### Define function for querying CPT data and metadata when a user selects a CPT test
```python
def on_test_widget_change(change):
   ax[0].lines.clear()
    ax[1].lines.clear()
    ax[2].lines.clear()
    if(change['new']!=-1):
        sql = 'SELECT SCPT.SCPT_DPTH, SCPT.SCPT_RES, SCPT.SCPT_FRES, SCPT.SCPT_PWP FROM SCPT INNER JOIN SCPG ON SCPT.SCPG_ID = SCPG.SCPG_ID WHERE SCPG.TEST_ID = ' + str(change['new'])
        scpt_df = ngl.read_sql(sql)
        line1, = ax[0].plot(scpt_df['SCPT_RES'].values,scpt_df['SCPT_DPTH'].values, c="C0")
        line2, = ax[1].plot(scpt_df['SCPT_FRES'].values,scpt_df['SCPT_DPTH'].values, c="C0")
        line3, = ax[2].plot(scpt_df['SCPT_PWP'].values,scpt_df['SCPT_DPTH'].values, c="C0")
        for a in ax:
            a.relim()
            a.autoscale_view(True)
        fig.canvas.draw()
        
        sql = 'SELECT SCPG.SCPG_CSA, SCPG.SCPG_RATE, SCPG.SCPG_CREW, SCPG.SCPG_METH, SCPG.SCPG_STAR, '
        sql += 'SCPG.SCPG_ENDD, SCPG.SCPG_PWP, SCPG.SCPG_REM FROM SCPG WHERE SCPG.TEST_ID = ' + str(change['new'])
        scpg_df = ngl.read_sql(sql)
        metadata = "<strong>CPT Test Metadata</strong><br>"
        metadata += "Cone area = " + str(scpg_df['SCPG_CSA'].values[0]) + ' cm<sup>2</sup><br>'
        metadata += "Push rate = " + str(scpg_df['SCPG_RATE'].values[0]) + ' cm/s<br>'
        metadata += "Crew = " + str(scpg_df['SCPG_CREW'].values[0]) + '<br>'
        metadata += "Method = " + str(scpg_df['SCPG_METH'].values[0]) + '<br>'
        metadata += "Start date = " + str(scpg_df['SCPG_STAR'].values[0]) + '<br>'
        metadata += "End date = " + str(scpg_df['SCPG_ENDD'].values[0]) + '<br>'
        metadata += "Position of pore pressure measurement = " + str(scpg_df['SCPG_PWP'].values[0]) + '<br>'
        metadata += "Remarks = " + str(scpg_df['SCPG_REM'].values[0]) + '<br>'
        metadata_widget.value = metadata
    else:
        ax[0].lines.clear()
        ax[1].lines.clear()
        ax[2].lines.clear()
        metadata_widget.value=''
    return
```

### Use the ipywidgets 'observe' command to link widgets to appropriate functions on change
```python
site_widget.observe(on_site_widget_change, names='value')
test_widget.observe(on_test_widget_change, names='value')
display(plot_widget)
display(metadata_widget)
```
