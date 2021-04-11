for ($num = 7 ; $num -le 12 ; $num++){
	Write-Host running Q1_$num
	& python .\Ex1.py --model-name "Q1_$num" --epochs 15
}