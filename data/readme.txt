This README.txt file was generated on 06.27.2024 by Robert Tamburo

#
# General instructions for completing README: 
# For sections that are non-applicable, mark as N/A (do not delete any sections). 
# Please leave all commented sections in README (do not delete any text). 
#

-------------------
GENERAL INFORMATION
-------------------

1. Title of Dataset:

#
# Authors: Include contact information for at least the 
# first author and corresponding author (if not the same), 
# specifically email address, phone number (optional, but preferred), and institution. 
# Contact information for all authors is preferred.
#

2. Author Information
<create a new entry for each additional author>

First Author Contact Information
    Name: Anurag Ghosh
    Institution: Carnegie Mellon University
    Address: NSH A401
    Email: anuraggh@andrew.cmu.edu
	Phone Number: NA


Corresponding Author Contact Information
    Name: Anurag Ghosh
    Institution: Carnegie Mellon University
    Address: NSH A401
    Email: anuraggh@andrew.cmu.edu
	Phone Number: NA

Author Contact Information (if applicable)
    Name: Robert Tamburo
    Institution: Carnegie Mellon University
    Address: NSH A401
    Email: rtamburo@andrew.cmu.edu
	Phone Number: NA

---------------------
DATA & FILE OVERVIEW
---------------------

#
# Directory of Files in Dataset: List and define the different 
# files included in the dataset. This serves as its table of 
# contents. 
#

Directory of Files:
   A. Filename: images.zip
      Short Description: Contains all the ROADWork images that have been manually annotated

   B. Filename: sem_seg_labels.zip
      Short Description: Contains semantic segmentation labels for images in images.zip in the Cityscapes format.

   C. Filename: annotations.zip
      Short Description: Contains instance segmentations, sign information, scene descriptions and other labels for images in images.zip in a COCO-like format. It contains multiple splits, suited for different tasks. Please see Usage for more information.

   D. Filename: discovered_images.zip
      Short Description: Contains discovered images with roadwork scenes from BDD100K and Mapillary dataset (less than 1000 images in total). These images are provided for ease of access ONLY. See below for specific license information for these external datasets.

   E. Filename: traj_images.zip
      Short Desription: Contains images associated with pathways. These images were manually filtered to contain ground truth pathways obtained from COLMAP. The split is described in Usage, to avoid data contamination from models trained on images.zip.

    F. Filename: traj_annotations.zip
       Short Description: Contains pathway annotations corresponding to images in traj_images.zip.
traj_images_dense.zip -- contains the dense set of images with associated pathways. These are similar to traj_images.zip, they are not subsampled.

    G. Filename: traj_annotations_dense.zip
       Short Description: Contains pathway annotations corresponding to images in traj_images_dense.zip


Additional Notes on File Relationships, Context, or Content 
(for example, if a user wants to reuse and/or cite your data, 
what information would you want them to know?):              


#
# File Naming Convention: Define your File Naming Convention 
# (FNC), the framework used for naming your files systematically 
# to describe what they contain, which could be combined with the
# Directory of Files. 
#

File Naming Convention: Based on description of contents.


#
# Data Description: A data description, dictionary, or codebook
# defines the variables and abbreviations used in a dataset. This
# information can be included in the README file, in a separate 
# file, or as part of the data file. If it is in a separate file
# or in the data file, explain where this information is located
# and ensure that it is accessible without specialized software.
# (We recommend using plain text files or tabular plain text CSV
# files exported from spreadsheet software.) 
#

-----------------------------------------
DATA DESCRIPTION FOR: [FILENAME]
-----------------------------------------
<create sections for each dataset included>


1. Number of variables: NA


2. Number of cases/rows: NA


3. Missing data codes: NA
        Code/symbol        Definition
        Code/symbol        Definition


4. Variable List

#
# Example. Name: Wall Type 
#     Description: The type of materials present in the wall type for housing surveys collected in the project.
#         1 = Brick
#         2 = Concrete blocks
#	  3 = Clay
#         4 = Steel panels


    A. Name: NA
       Description: NA
                    Value labels if appropriate


    B. Name: NA
       Description: NA
                    Value labels if appropriate

--------------------------
METHODOLOGICAL INFORMATION
--------------------------

#
# Software: If specialized software(s) generated your data or
# are necessary to interpret it, please provide for each (if
# applicable): software name, version, system requirements,
# and developer. 
#If you developed the software, please provide (if applicable): 
#A copy of the software’s binary executable compatible with the system requirements described above. 
#A source snapshot or distribution if the source code is not stored in a publicly available online repository.
#All software source components, including pointers to source(s) for third-party components (if any)

1. Software-specific information:
<create a new entry for each qualifying software program>

Name: TBD
Version: TBD
System Requirements: TBD
Open Source? (Y/N): TBD

(if available and applicable)
Executable URL: TBD
Source Repository URL: TBD
Developer: TBD
Product URL: TBD
Software source components: TBD


Additional Notes(such as, will this software not run on 
certain operating systems?): Software should be cross platform compatible. See code information in Usage section of description.


#
# Equipment: If specialized equipment generated your data,
# please provide for each (if applicable): equipment name,
# manufacturer, model, and calibration information. Be sure
# to include specialized file format information in the data
# dictionary.
#

2. Equipment-specific information: NA
<create a new entry for each qualifying piece of equipment>

Manufacturer: NA
Model: NA

(if applicable)
Embedded Software / Firmware Name: NA
Embedded Software / Firmware Version: NA
Additional Notes: NA

#
# Dates of Data Collection: List the dates and/or times of
# data collection.
#

3. Date of data collection (single date, range, approximate date) <suggested format YYYYMMDD>: Data collection started around 20210601 and ended in 2024

