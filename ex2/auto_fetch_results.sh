# #!/bin/sh

SLEEP_TIME=1

while [ true ]
do
    # sshpass -p aso1076a rsync -a --progress -u --prune-empty-dirs samsona@urim.md.biu.ac.il:/home/samson-vol/samsona/itamar/ex2/Results/May-28_03.07.22_g-M11_d-M12_l-NonSaturated_e-50_k-4_b-8_z-100_d-celeb_gs-3_ds-1_name-urim_glr-0.0004_dlr-0.0001/ ./Results/May-28_03.07.22_g-M11_d-M12_l-NonSaturated_e-50_k-4_b-8_z-100_d-celeb_gs-3_ds-1_name-urim_glr-0.0004_dlr-0.0001/
    sshpass -p aso1076a rsync -a --progress -u --prune-empty-dirs samsona@urim.md.biu.ac.il:/home/samson-vol/samsona/itamar/NN4Images/ex2//Results/ ./Results/
    rsync -a --progress -u --prune-empty-dirs  ./Results/ /media/itamar/Win-Linux/Results/NN4Images/
    dt=$(date '+%H:%M:%S');
    echo
    echo "$dt" ": Finished fetching."
    echo "$dt" ": Now sleeping for $SLEEP_TIME minutes... enter ctrl+c to exit"
    echo
    sleep "$SLEEP_TIME"m
done