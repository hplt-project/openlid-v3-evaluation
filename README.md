# OpenLID-v3: Precision-Driven Language Identification Tool

This repository contains data and code to reproduce evaluations, described in the paper "OpenLID-v3: Precision-Driven Language Identification Tool", accepted to [VarDial 2026](https://sites.google.com/view/vardial-2026/) workshop, collocated with [EACL 2026](https://2026.eacl.org/).

For training the OpenLID-v3 model and running it for inference, refer to [its repository](https://github.com/hplt-project/openlid).

## Adding GlotLID data

The data on LUMI are in `/scratch/project_465001890/eurolid/glotlid-corpus/`.

It is also possible download them using the script `download_glotlid.py`.

`make_list_of_glotlid_sources.py` creates the list of GlotLID sources for each language and shows number of samples in GlotLID data.
There is no need to run it, since the resulting list is in `other.tsv` in the root of this repository.

The script `add_from_glotlid.py` shows how to select only the data sources that are of reliable quality and not proprietary. (Beware of hardcoded paths...)
The list of filters there is also for the languages we worked with before;
for Scandinavian etc., if there are some other sources, check their quality and license according to [GlotLID list](https://github.com/cisnlp/GlotLID/blob/main/sources.md).
We also collected licenses of the sources we used [here](https://docs.google.com/spreadsheets/d/162EzUGXDllmujoNG5s_XngSlL4awOJ9F79t5k2OM_FQ/edit?gid=737547198#gid=737547198) at LangID sources sheet.

That script also ensures that wikipedia GlotLID data do not intersect with OpenLID wikipedia data.

---------------------------------------------------------------------------------------------

<sub><sup>This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546].
The contents of this publication are the sole responsibility of the HPLT consortium and do not necessarily reflect the opinion of the European Union.</sup></sub>