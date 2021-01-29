# DeepEPI

This repository contains the scripts, data, and trained models for DeepEPI. DeepEPI is a deep learning model to predict enhancer-promoter interactions by employing features from large genomic contexts.

# Requirements

* numpy
* scikit-learn
* PyTorch>=1.6.0

# Data preparation

## Genomic data
    - CTCF data in narrowPeak  
    - DNase-seq data in bigWig (p-value track)
    - H3K27ac ChIP-seq data in bigWig (p-value track)
    - H3K4me3 ChIP-seq data in bigWig (p-value track)
    - H3K4me1 ChIP-seq data in bigWig (p-value track)
The users should edit the json files in `config/` to specify the location of these genomic data.


# Usage

```bash
./run_deepepi.sh /path/to/query/file /path/to/output/directory
```

## Input

The input file should be formatted as:

```
##CELLTYPE [cell type]
##MODEL [model name]
#chrom enhancer promoter
chr10 49875920-49876712 50396056-50398056
chr10	49874816-49877816	49874816-49877816
```


## Output
The output will be saved at `/path/to/output/directory/results.txt`

## Demo

```bash
./run_deepepi.sh demo/demo_data.tsv output/demo
```

For questions about the datasets and code, please contact [chenkenbio@gmail.com](mailto:chenkenbio@gmail.com).

