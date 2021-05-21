## Query Site Table Using Pandas

An easy way to query the database is to use the Pandas read_sql command, which queries data and returns a Pandas dataframe. 
The commands below imports the Pandas and ngl_db packages, creates a connection object to ngl_db called cnx, creates a 
string called sql that queries all information from the SITE table, and creates a Pandas dataframe called df that contains 
the results of the query.

```python
import pandas as pd
import ngl_db

cnx = ngl_db.connect()
sql = "SELECT * FROM SITE"
df = pd.read_sql(sql,cnx)
df
```

The output from the command is illustrated in the figure below. When this query was written, there were a total of 333 sites in 
the NGL database. The SITE_ID field is not contiguous because sites are sometimes deleted from the database, and the 
SITE_ID field is never re-used. The Pandas dataframe is broken between SITE_ID 151 and 677 for ease of displaying 
information in the output window. Many rows of data are not displayed in Figure 2 as a result.

<img src="SiteTableQuery.png" alt="Screenshot of results from site table query">
