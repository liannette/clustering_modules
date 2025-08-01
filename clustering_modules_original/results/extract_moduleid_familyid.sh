#!/bin/bash

for i in $(seq 600 200 1800); do
    input_file="PRESTO-STAT_modules_kmeans_${i}_families.txt"
    output_file="stat_module_${i}_families.txt"
    awk '{print $1, $7}' "$input_file" > "$output_file"
    sed -i '1s/.*/module_id\tmodule_family_id/' "$output_file"
done

# for i in $(seq 2000 2000 20000); do
#     input_file="PRESTO-STAT_modules_kmeans_${i}_families.txt"
#     output_file="stat_module_${i}_families.txt"
#     awk '{print $1, $7}' "$input_file" > "$output_file"
#     sed -i '1s/.*/module_id\tmodule_family_id/' "$output_file"
# done