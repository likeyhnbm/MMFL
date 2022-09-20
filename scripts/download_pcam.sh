FILENAME="dataset/pcam/pcam/camelyonpatch_level_2_split_train_x.h5.gz"
FILEID="1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
FILENAME="dataset/pcam/pcam/camelyonpatch_level_2_split_train_y.h5.gz"
FILEID="1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
FILENAME="dataset/pcam/pcam/camelyonpatch_level_2_split_test_x.h5.gz"
FILEID="1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
FILENAME="dataset/pcam/pcam/camelyonpatch_level_2_split_test_y.h5.gz"
FILEID="17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && rm -rf /tmp/cookies.txt
gunzip dataset/pcam/pcam/*.gz