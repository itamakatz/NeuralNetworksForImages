for ($num = 13 ; $num -le 18 ; $num++){
	Write-Host running Q1_$num
	& python ..\Ex1.py --model-name "Q1_$num" --epochs 15
}