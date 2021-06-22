# #!/bin/sh
cd ..

# python3 ./ex2.py -g "M1" -d "M1" -b 100 -l "non_saturated" -k 0 -e 50 -z 150 --dataset "celeb"
# python3 ./ex2.py -g "M11" -d "M12" -b 128 -l "non_saturated" -k 2 -e 50 -z 100 -gs 5 -ds 2 --dataset "celeb" -sm

GEN="M1"
DIS="M3"

# unrolling steps
K_MIN=0
K_MAX=4
K_STEP=2

GS_MIN=1
GS_MAX=5
GS_STEP=2

DS_MIN=1
DS_MAX=5
DS_STEP=2

# epochs
E_MIN=5
E_MAX=5
E_STEP=1

# latent space
Z_MIN=100
Z_MAX=100
Z_STEP=1

BATCH_MIN=48
BATCH_MAX=48
BATCH_STEP=1

# LOSS=( "H" "mse" "non_saturated" )
# LOSS=( "mse" )
# LOSS=( "non_saturated" )
LOSS=( "H" )

if [[ "run" == $1 || "print" == $1 ]]; then
	
	COUNT=0
	for ((e=$E_MIN;e<=$E_MAX;e=e+$E_STEP)); do
	for ((z=$Z_MIN;z<=$Z_MAX;z=z+$Z_STEP)); do
	for ((b=$BATCH_MIN;b<=$BATCH_MAX;b=b+$BATCH_STEP)); do
	for ((k=$K_MIN;k<=$K_MAX;k=k+$K_STEP)); do
    for ((gs=$GS_MIN;gs<=$GS_MAX;gs=gs+$GS_STEP)); do
    for ((ds=$DS_MIN;ds<=$DS_MAX;ds=ds+$DS_STEP)); do
	for l in "${LOSS[@]}"; do

        # if [ "$gs" -lt "$ds" ]; then
        #     continue
        # fi

		COUNT=$((COUNT + 1))
		if [ "run" = $1 ]; then
			python3 ./ex2.py -g $GEN -d $DIS -b $b -l $l -e $e -z $z --dataset "MNIST" -k $k -gs $gs -ds $ds
		fi
		if [ "print" = $1 ]; then
			echo python3 ./ex2.py -g \"$GEN\" -d \"$DIS\" -b $b -l \"$l\" -e $e -z $z --dataset "MNIST" -k $k -gs $gs -ds $ds
		fi
	done
	done
	done
	done
    done
    done
	done

	echo Overall $COUNT executions
else
	echo Please enter either \'run\' or \'print\' as a single parameter
fi