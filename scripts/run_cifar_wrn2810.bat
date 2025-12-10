@echo off
REM Train WideResNet-28-10 on CIFAR-100 (Windows Batch Script)

echo Training WideResNet-28-10 on CIFAR-100...
python src\train_cls.py --config configs\cifar100_wrn2810.yaml

echo.
echo Evaluating best model...
python src\eval_cls.py --config configs\cifar100_wrn2810.yaml --checkpoint runs\cifar_wrn2810\best_model.pth --split test --save-confusion

echo.
echo Training complete! Check runs\cifar_wrn2810\ for results.
pause
