# FILENAME="head.pt"
# FILEID="1S28x9XK7x2aswTwWjMWVl0wawm5ZqwGk"
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/1M0FYASFZbHNsv4fdOgqIpYaSpc_mQyHK/view?usp=sharing
# https://drive.google.com/file/d/1w8dfSzpQ9MVOs-b3B_KhHaN2HWmz9jVe/view?usp=sharing
# https://drive.google.com/file/d/1S28x9XK7x2aswTwWjMWVl0wawm5ZqwGk/view?usp=sharing
# https://drive.google.com/file/d/1U0wz7jNgfEN0eGtq4m3i848A27LXBxOJ/view?usp=sharing
# https://drive.google.com/file/d/1QqnCLdM2Z5H3dBS2x2oLmdbDwICu4a17/view?usp=sharing
# https://drive.google.com/file/d/1S28x9XK7x2aswTwWjMWVl0wawm5ZqwGk/view?usp=sharing

# python compute_hessian.py --model_dir /home/guangyu/FedPrompt/bias.pt --test
# python compute_hessian.py --model_dir /home/guangyu/FedPrompt/prompt.pt --test
# python compute_hessian.py --model_dir /home/guangyu/FedPrompt/pretrain.pt --test
# python compute_hessian.py --model_dir /home/guangyu/FedPrompt/adapter.pt --test
python compute_hessian2.py --model_dir /home/guangyu/FedPrompt/head.pt --test