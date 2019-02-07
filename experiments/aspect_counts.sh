pushd ../perspective

docify.py -i ../data/raw/kaggle1 -o ../data/cache/kaggle1_docs.json


pushd aspect_detection/bootstrap

echo "---- 1 ----"
detection.py -i ../../../data/cache/kaggle1_docs.json -c 1
echo "---- 10 ----"
detection.py -i ../../../data/cache/kaggle1_docs.json -c 10
echo "---- 100 ----"
detection.py -i ../../../data/cache/kaggle1_docs.json -c 100
echo "---- 1000 ----"
detection.py -i ../../../data/cache/kaggle1_docs.json -c 1000
echo "---- 10000 ----"
detection.py -i ../../../data/cache/kaggle1_docs.json -c 10000

popd

popd
