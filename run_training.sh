until python3 train.py --resume 0 --train_dir dataset2/train/ --test_dir dataset2/test --batch_size 64 --epochs 100 --model 3 --lr 2> training_log.log 2>&1 & disown; do
    echo "system crashed : error : $?, respawning... " &> crash.log
    sleep 1
done

until tensorboard --logdir=logs/ --port 5252 > /dev/null 2>&1 & disown; do
    echo "tensorboard crash..." >&2

    sleep 1
done
