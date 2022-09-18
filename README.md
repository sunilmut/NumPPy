# Setup:
## Install git for windows:
In a browser go [here](https://git-scm.com/download/win), download and
run the "64-bit Git for Windows Setup"<br/>
Or<br/>
Open a administrator command prompt and run:

```
winget install --id Git.Git -e --source winget
```

## Clone the code:
Open a new command prompt and run:

```
git clone https://github.com/sunilmut/NumPPy.git
cd NumPPy
```

## Install python
Windows:
Follow the instructions [here](https://docs.microsoft.com/en-us/windows/python/scripting) to install python
on your system

## Install the necessary python modules:
In the opened command prompt, run:

```
pip3 install guizero numpy pandas xlrd xlsxwriter matplotlib
```

## Update the code:
If you have to update the code, for example to pull in a fix or an update.<br/>
**Note**
This assumes that you have already done the setup.<br/>

Open a command prompt:
```
cd NumPPy
git pull
```

# Projects:
## Run FreezeData processing app
In the opened command prompt (or open a command prompt), run:
```
python binary.py
```

To run the app in verbose mode (to get more logs), run:
```
python binary.py -v
```

## XLS (Excel) color sorter
Sorts the various colors in a given input xls file and outputs
them to a xlsx file.<br/>
It can parse and combine color codes up to 100 different sheets
within a xls workbook.

### Example:
Input file:
![plot](./colorsort_input1.png width="400" height="400")

### Usage:
```
python colorsort.py
```
To get verbose output log fie:
```
python colorsort.py -v
```

**Note**
Only works with xls files for input for now.<br/>
Close the output file prior to running.
