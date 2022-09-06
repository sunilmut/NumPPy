# Setup:
## Install git for windows:
In a browser go to https://git-scm.com/download/win, download and run the "64-bit Git for Windows Setup"
or
Open a administrator command prompt and run:
`winget install --id Git.Git -e --source winget`

## Clone the code:
Open a new command prompt and run:
`git clone https://github.com/sunilmut/xls_format_parser.git`
`cd xls_format_parser`

## Install python
Windows:
Follow the instructions [here](https://docs.microsoft.com/en-us/windows/python/scripting) to install python
on your system

# For running the freeze.py script:
## Install the necessary python modules:
In the opened command prompt, run:
`pip3 install guizero numpy pandas`

## Run the code:
In the opened command prompt, run:
`python freeze.py`

# xls_format_parser
Parses formatted xls data based on the color code of the columns
and outputs an xls file that is differentiated based on the color
code.
It can parse and combine color codes up to 100 different sheets
within a xls workbook.

Usage:
parse_xls.py -i <inputfile> -o <outputfile>'
only works with xls files for input for now.
close the output file prior to running.
