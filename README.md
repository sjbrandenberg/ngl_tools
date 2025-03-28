# ngl_tools
This is a collection of tools developed by the Next Generation Liquefaction modelers. The Next Generation Liquefaction project includes a relational database of liquefaction case histories from earthquakes around the world, which is accessible at [https://nextgenerationliquefaction.org/](https://nextgenerationliquefaction.org/). The project also includes efforts to develop models from the new data that is included in the database, and supporting studies that focus on knowledge gaps. This repository contains code that has been used by the NGL modeling teams to develop their models. 

## installation

```pip install ngl_tools```  

## smt
The NGL Supported Modeling Team (SMT) members are Kristin J. Ulmer, Kenneth L. Hudson, Scott J. Brandenberg, Jonathan P. Stewart, Paolo Zimmaro, and Steven L. Kramer. We developed tools that process cone penetration test data, including inverse-filtering to remove thin layer effects following Boulanger and DeJong (2018), an agglomerative clustering algorithm by Hudson et al. (2024) to identify stratigraphic layers from inverse-filtered cone data, and a number of additional helper functions to compute soil behavior type index, overburden- and fines-corrected cone tip resistance, etc. Details of these codes can be found in the wiki associated with this repository. 
