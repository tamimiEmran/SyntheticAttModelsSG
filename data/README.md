# Dataset Setup Guide

## Prerequisites
- You may use wget and unzip or download the files yourself

## SGCC Dataset

#### 1. Download split archives
data.zip:
```bash
https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip
```
data.z01
```bash
https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01
```
data.z02
```bash
https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02
```
#### 2. Extract all three files to a single csv file at
````bash
data/raw/data.csv
````

# Ausgrid Dataset

## 1. Download all yearly datasets from
### 1 July 2010 to 30 June 2011:
```bash
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip
```
#### Extract the file to
```bash
data/raw/2010-2011 Solar home electricity data.csv
```

### 1 July 2011 to 30 June 2012:
```bash
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip
```
#### Extract the file to:
```bash
data/raw/2011-2012 Solar home electricity data v2.csv
```

### 1 July 2012 to 30 June 2013:
```bash
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip
```
#### Extract the file to:
```bash
data/raw/2012-2013 Solar home electricity data v2.csv
```


## Final Expected Structure
```bash
data/raw/
├── data.csv
├── 2010-2011 Solar home electricity data.csv
├── 2011-2012 Solar home electricity data v2.csv
└── 2012-2013 Solar home electricity data v2.csv
```