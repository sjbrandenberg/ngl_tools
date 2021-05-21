# Background Info

## Connecting To Database

Connecting to a relational database requires credentials, like username, password, database name, and hostname. 
Rather than requiring users to know these credentials, we have created a Python package that allows users to 
connect to the database. The lines of code below first imports the ngl_db Python package, and then creates a 
connection object to the database called cnx.

```python
import ngl_db
cnx = ngl_db.connect()
```

## Understanding the database schema

The NGL database is organized into tables that are related to each other via keys. To query the database, 
you will need to understand the organizational structure of the database, called the schema. The database 
schema is documented at the following URL:

[https://nextgenerationliquefaction.org/schema/index.html](https://nextgenerationliquefaction.org/schema/index.html)

Figure 1 describes the schema for the SITE table, which is a high level table in the NGL database where 
users enter information about a particular site they have investigated following an earthquake. The SITE 
table contains SITE_ID, which is the primary key for the SITE table. Every entry in the SITE table is assigned 
a unique SITE_ID that identifies the entry. Additional fields include SITE_NAME, SITE_LAT, SITE_LON, SITE_GEOL, 
SITE_REM, SITE_STAT, and SITE_REVW.  The Children column in Figure 1 identifies other tables in the NGL 
database that have been assigned a foreign key constraint to the SITE_ID field. For example, FLDO is a table 
containing field observations of liquefaction at a site. The FLDO table has a SITE_ID field, called a foreign 
key, that identifies the observation as being associated with the site with the same SITE_ID.

![](SiteSchema.png)
<strong>Figure 1.</strong> Screenshot of NGL site table schema.

## Notebooks published in DesignSafe

Brandenberg, S. J. , and Zimmaro, P. (2019). "'Next Generation Liquefaction (NGL) Partner Dataset - Sample Queries', in Next Generation Liquefaction (NGL) Partner Dataset - Sample Queries DesignSafe-CI, 10.17603/ds2-xvp9-ag60 [link](https://doi.org/10.17603/ds2-xvp9-ag60)

Brandenberg, S. J. , and Zimmaro, P. (2019). "'Next Generation Liquefaction (NGL) Partner Dataset Cone Penetration Test (CPT) Viewer', in Next Generation Liquefaction (NGL) Partner Dataset Cone Penetration Test (CPT) Viewer DesignSafe-CI, 10.17603/ds2-99kp-rw11 [link](https://doi.org/10.17603/ds2-99kp-rw11)

Brandenberg, S. J. , and Zimmaro, P. (2019). "'Next Generation Liquefaction (NGL) Partner Dataset - Surface Wave Viewer', in Next Generation Liquefaction (NGL) Partner Dataset - Surface Wave Viewer. DesignSafe-CI, 10.17603/ds2-cmn0-h864 [link](https://doi.org/10.17603/ds2-cmn0-h864)

Zimmaro, P. , and Brandenberg, S. J. (2019). "'Next Generation Liquefaction (NGL) Partner Dataset - Invasive Geophysical Test Viewer', in Next Generation Liquefaction (NGL) Partner Dataset - Invasive Geophysical Test Viewer. DesignSafe-CI, 10.17603/ds2-tq39-kp49 [link](https://doi.org/10.17603/ds2-tq39-kp49)

Lee, A. , Fisher, H. , Zimmaro, P. , and Brandenberg, S. J. (2019). "'Next Generation Liquefaction (NGL) Partner Dataset - Boring Log Viewer', in Next Generation Liquefaction (NGL) Partner Dataset - Boring Log Viewer. DesignSafe-CI, 10.17603/ds2-sj7t-av93 [link](https://doi.org/10.17603/ds2-sj7t-av93)

Brandenberg, S. J. , Zimmaro, P. , Lee, A. , Fisher, H. , and Stewart, J. P. (2019). "'Next Generation Liquefaction (NGL) Partner Dataset', in Next Generation Liquefaction (NGL) Partner Dataset DesignSafe-CI, 10.17603/ds2-2xzy-1y96 [link](https://doi.org/10.17603/ds2-2xzy-1y96)
