```markdown
# Dataset Setup Guide

## Prerequisites
- You may use wget and unzip or download the files from the GUI

## SGCC Dataset
```bash
# Download split archives
wget -P data/raw \
     https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.zip \
     https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z01 \
     https://github.com/henryRDlab/ElectricityTheftDetection/raw/master/data.z02

# Merge & extract (Linux/Mac)
zip -s 0 data.zip --out merged.zip && unzip merged.zip -d data/raw

# Windows users: Use 7-Zip GUI or:
unzip data.zip -d data/raw/

```

## Ausgrid Dataset
```bash
# Download all yearly datasets
wget -P data/raw \
  https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2010-to-30-June-2011.zip \
  https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2011-to-30-June-2012.zip \
  https://www.ausgrid.com.au/-/media/Documents/Data-to-share/Solar-home-electricity-data/Solar-home-half-hour-data---1-July-2012-to-30-June-2013.zip

# Extract and rename (run from project root)
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
```