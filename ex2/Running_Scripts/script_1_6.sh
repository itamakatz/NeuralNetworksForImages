# #!/bin/sh
cd ..

python3 ./ex2.py -g "M1" -d "M3" -b 48 -l "H" -k 0 -e 50 -z 50 -gs 1 -ds 3 --dataset "MNIST" -sm
echo "==========================================="
echo "==========================================="
python3 ./ex2.py -g "M1" -d "M3" -b 48 -l "H" -k 0 -e 50 -z 50 -gs 1 -ds 1 --dataset "MNIST" -sm
echo "==========================================="
echo "==========================================="
python3 ./ex2.py -g "M1" -d "M3" -b 48 -l "H" -k 4 -e 50 -z 50 -gs 1 -ds 1 --dataset "MNIST" -sm