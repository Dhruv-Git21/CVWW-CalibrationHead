@echo off
REM Train ViT-Tiny on CIFAR-100 (Windows Batch Script)

echo Training ViT-Tiny on CIFAR-100...
python src\train_cls.py --config configs\cifar100_vit_tiny.yaml

echo.
echo Evaluating best model...
python src\eval_cls.py --config configs\cifar100_vit_tiny.yaml --checkpoint runs\cifar_vit_tiny\best_model.pth --split test --save-confusion

echo.
echo Training complete! Check runs\cifar_vit_tiny\ for results.
pause
