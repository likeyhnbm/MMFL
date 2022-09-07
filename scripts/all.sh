# Standard setting
# sh scripts/c16_r_1_alpha_0.5/prompt.sh
sh scripts/c16_r_1_alpha_0.5/pretrain.sh
# More client Do it later
# sh scripts/c64_r_1_alpha_0.5/prompt.sh
# sh scripts/c64_r_1_alpha_0.5/pretrain.sh

# Sampling
# sh scripts/c64_r_0.25_alpha_0.5/prompt.sh
sh scripts/c64_r_0.25_alpha_0.5/pretrain.sh

# Sampling less
# sh scripts/c64_r_0.125_alpha_0.5/prompt.sh
# sh scripts/c64_r_0.125_alpha_0.5/pretrain.sh

# Reduce alpha
# Standard setting
# sh scripts/c16_r_1_alpha_0.1/prompt.sh
sh scripts/c16_r_1_alpha_0.1/pretrain.sh
# More client Do it later
# sh scripts/c64_r_1_alpha_0.1/prompt.sh
# sh scripts/c64_r_1_alpha_0.1/pretrain.sh

# Sampling
# sh scripts/c64_r_0.25_alpha_0.1/prompt.sh
# sh scripts/c64_r_0.25_alpha_0.1/pretrain.sh

# # Sampling less
# sh scripts/c64_r_0.125_alpha_0.1/prompt.sh
# sh scripts/c64_r_0.125_alpha_0.1/pretrain.sh

sh scripts/re-test.sh
python scripts/homo.py