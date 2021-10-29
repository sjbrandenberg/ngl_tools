========================================
Connecting to the database in DesignSafe
========================================

Connecting to a relational database requires credentials, like username, password, database name, and hostname. 
Rather than requiring users to know these credentials, we have created a Python package that allows users to 
connect to the database. The lines of code below first imports the ngl_db Python package, and then creates a 
connection object to the database called cnx.

.. code-block:: python

  import ngl_db
  cnx = ngl_db.connect()
