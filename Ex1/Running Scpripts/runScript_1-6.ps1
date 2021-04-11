for ($num = 1 ; $num -le 6 ; $num++){
	Write-Host running Q1_$num
	& python ..\Ex1.py --model-name "Q1_$num" --epochs 15
}