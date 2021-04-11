for ($num = 19 ; $num -le 22 ; $num++){
	Write-Host running Q1_$num
	& python ..\Ex1.py --model-name "Q1_$num" --epochs 15
}