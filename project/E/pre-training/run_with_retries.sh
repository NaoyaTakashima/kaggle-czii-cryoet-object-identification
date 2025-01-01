#!/bin/bash

max_retries=10  # 最大リトライ回数
count=0

# コマンドを引数として受け取る
cmd="$@"

if [ -z "$cmd" ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

while [ $count -lt $max_retries ]; do
    eval $cmd
    if [ $? -eq 0 ]; then
        echo "Execution successful."
        break
    else
        echo "Execution failed. Retrying... ($((count+1))/$max_retries)"
        count=$((count+1))
        sleep 2  # 再試行前の待機時間
    fi
done

if [ $count -eq $max_retries ]; then
    echo "Max retries reached. Exiting."
    exit 1
fi
