# Dataset Setup Guide

## Prerequisites
- You may use wget and unzip or download the files yourself

## SGCC Dataset

#### 1. Download split archives
```bash
wget -P data/raw \
     https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip \
     https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01 \
     https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02
```
#### or download files from
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
#### 2. Merge & extract (Linux/Mac)
```bash
zip -s 0 data.zip --out merged.zip && unzip merged.zip -d data/raw
```
#### 2. Windows users: Use any GUI or:
```bash
unzip data.zip -d data/raw/
```

## Ausgrid Dataset

### 1. Download all yearly datasets
```bash
wget -P data/raw \
  https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip \
  https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip \
  https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip
```
#### or download files from
1 July 2010 to 30 June 2011:
```bash
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip
```
1 July 2011 to 30 June 2012:
```bash
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip
```
1 July 2012 to 30 June 2013:
```bash
https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip
```
# Extract and rename (run from project root)
```bash
cd data/raw && \
unzip "Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip" && \
mv "Solar home electricity data.csv" "2010-2011 Solar home electricity data.csv" && \
unzip "Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip" && \
mv "Solar home electricity data v2.csv" "2011-2012 Solar home electricity data v2.csv" && \
unzip "Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip" && \
mv "Solar home electricity data v2.csv" "2012-2013 Solar home electricity data v2.csv" && \
cd ../..
```

## Final Structure
```bash
data/raw/
├── data.csv
├── 2010-2011 Solar home electricity data.csv
├── 2011-2012 Solar home electricity data v2.csv
└── 2012-2013 Solar home electricity data v2.csv
```
> **Note**: Commands assume execution from repository root. Filename renames ensure compatibility with processing scripts.
