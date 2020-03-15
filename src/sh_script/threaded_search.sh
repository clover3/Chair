#!/bin/bash
#SBATCH --job-name=galago_search
#SBATCH --output=/mnt/nfs/work3/youngwookim/data/log/%j.log
#SBATCH --mem=4gb

galago threaded-batch-search --index=${index_path} --requested=500 ${query_file} > ${outpath}
