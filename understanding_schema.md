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

<img src="https://github.com/sjbrandenberg/ngl_tools/blob/master/SiteSchema.png" alt="Screenshot of site table schema">
<strong>Figure 1.</strong> Screenshot of NGL site table schema.
