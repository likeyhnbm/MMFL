# client_num client_sample partition_method alpha thread_num
sh scripts/custom/adapter 16 1 hetero 0.5 8
sh scripts/custom/adapter 16 1 hetero 0.1 8
sh scripts/custom/adapter 64 0.125 hetero 0.5 8
sh scripts/custom/adapter 64 0.125 hetero 0.1 8
sh scripts/custom/adapter 16 1 homo 0.5 8
sh scripts/custom/adapter 64 0.125 homo 0.5 8