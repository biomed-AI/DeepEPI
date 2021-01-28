#!/bin/bash

# if [ $# -lt 2 ]; then
#     echo "usage: $0 input outdir"
#     exit 1
# fi

input="$1"
outdir="$2"

input="/home/chenken/Documents/DeepEPI/demo/demo_data.tsv"
outdir="/home/chenken/Documents/DeepEPI/output/demo"

basedir=$(dirname `realpath $0`)


$basedir/src/format_file.py $input -o $outdir

name=`basename ${outdir}/*_enhancer.bed | cut -d '_' -f 1`
echo $name
source config/list.sh
${basedir}/src/prepare_data.py -d $outdir \
   --name $name \
   --metadata \
   --segment-dist --segment-mark \
   --segment-bigwig $dnase_config $h3k27ac_config $h3k4me1_config $h3k4me3_config \
   --segment-bed $ctcf_config \
   --segment-dist -t 4

${basedir}/src/predict.py -t $outdir  \
    -m ./model/GM12878_IMR90_K562_NHEK_chr1_19.model \
    -c ./model/GM12878_IMR90_K562_NHEK_chr1_19.config

# usage: prepare_data.py [-h] -d DATA_DIR --name NAME [--metadata]
#                        [--segment-bigwig SEGMENT_BIGWIG [SEGMENT_BIGWIG ...]]
#                        [--segment-bed SEGMENT_BED [SEGMENT_BED ...]]
#                        [--segment-mark] [--segment-dist] [-t NTHREADS]
# prepare_data.py: error: the following arguments are required: -d/--data_dir, --name
