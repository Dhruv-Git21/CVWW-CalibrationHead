@echo off
REM Train ResNet-50 on CIFAR-100 (Windows Batch Script)

echo Training ResNet-50 on CIFAR-100...
python src\train_cls.py --config configs\cifar100_resnet50.yaml

echo.
echo Evaluating best model...
python src\eval_cls.py --config configs\cifar100_resnet50.yaml --checkpoint runs\cifar_resnet50\best_model.pth --split test --save-confusion

echo.
echo Training complete! Check runs\cifar_resnet50\ for results.
pause
