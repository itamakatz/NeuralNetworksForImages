# #!/bin/sh
cd ..

# python3 ./ex2.py -g "M1" -d "M3" -b 48 -l "non_saturated" -k 4 -e 50 -z 50 --data="MNIST"

GEN="M1"
DIS="M3"

# unrolling steps
K_MIN=6
K_MAX=6
K_STEP=2

# epochs
E_MIN=100
E_MAX=100
E_STEP=1

# latent space
Z_MIN=50
Z_MAX=50
Z_STEP=40

BATCH_MIN=150
BATCH_MAX=150
BATCH_STEP=24

# LOSS=( "H" "mse" "non_saturated" )
# LOSS=( "mse" )
LOSS=( "non_saturated" )
# LOSS=( "H" )

if [[ "run" == $1 || "print" == $1 ]]; then
	
	COUNT=0
	for ((e=$E_MIN;e<=$E_MAX;e=e+$E_STEP)); do
	for ((z=$Z_MIN;z<=$Z_MAX;z=z+$Z_STEP)); do
	for ((b=$BATCH_MIN;b<=$BATCH_MAX;b=b+$BATCH_STEP)); do
	for ((k=$K_MIN;k<=$K_MAX;k=k+$K_STEP)); do
	for l in "${LOSS[@]}"; do

		COUNT=$((COUNT + 1))
		if [ "run" = $1 ]; then
			python3 ./ex2.py -g $GEN -d $DIS -b $b -l $l -k $k -e $e -z $z		
		fi
		if [ "print" = $1 ]; then
			echo python3 ./ex2.py -g \"$GEN\" -d \"$DIS\" -b $b -l \"$l\" -k $k -e $e -z $z
		fi
	done
	done
	done
	done
	done

	echo Overall $COUNT executions
else
	echo Please enter either \'run\' or \'print\' as a single parameter
fi