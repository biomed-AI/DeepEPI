# DeepEPI

This repository contains the scripts, data, and trained models for DeepEPI. DeepEPI is a deep learning model to predict enhancer-promoter interactions by employing features from large genomic contexts.

# Requirements

* numpy
* scikit-learn
* PyTorch>=1.6.0

# Data preparation

- CTCF data in narrowPeak  
- DNase-seq data in bigWig
- H3K27ac ChIP-seq data in bigWig
- H3K4me3 ChIP-seq data in bigWig
- H3K4me1 ChIP-seq data in bigWig

# Usage

```bash
./run_deepepi.sh $input $outdir
```
## Input


## Output
The output will be saved at `$outdir/results`


## Demo

```bash
./run_deepepi.sh demo/demo_data.tsv output/demo
```


For questions about the datasets and code, please contact [chenkenbio@gmail.com](mailto:chenkenbio@gmail.com).

