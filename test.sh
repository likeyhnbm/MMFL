for lr in 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
do
    if [ "$lr" != "1e-5" ];
    then
        echo "abc" $lr
    fi
    if [ "$lr" != "1e-3" ];
    then
        echo "123" $lr
    fi

done