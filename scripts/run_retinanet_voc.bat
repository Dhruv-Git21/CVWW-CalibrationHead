@echo off
REM Train RetinaNet on Pascal VOC (Windows Batch Script)

echo Training RetinaNet on Pascal VOC...
echo Note: Make sure Pascal VOC dataset is downloaded to data\VOCdevkit\
echo.

python src\train_det.py --config configs\detector_retinanet_voc.yaml

echo.
echo Evaluating best model...
python src\eval_det.py --config configs\detector_retinanet_voc.yaml --checkpoint runs\retinanet_voc\best_model.pth

echo.
echo Training complete! Check runs\retinanet_voc\ for results.
pause
