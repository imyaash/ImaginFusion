:: Coke bottle (takes about 90 minutes)
python cli.py "classic coca cola bottle" "Coke2" --fp16 --seed 0 --iters 10000 --lr 7.75e-4 --datasetSizeTrain 50 --sdVersion "1.5"
:: Coke bottle (takes about 30 minutes)
python cli.py "classic coca cola bottle" "Coke" --fp16 --seed 0 --iters 5000 --lr 7.75e-4 --datasetSizeTrain 100 --sdVersion "1.5"