#/bin/bash

set -e

# Only set for ensembles. Otherwise ignored.
BASE_IMPLEMENTATION=""

for i in "$@"
do
case $i in
    --outpath=*)
    OUTPATH="${i#*=}"
    shift # past argument=value
    ;;
    --type=*)
    TYPE="${i#*=}"
    shift # past argument=value
    ;;
    --modelpath=*)
    MODELPATH="${i#*=}"
    shift # past argument=value
    ;;
    --modelname=*)
    MODELNAME="${i#*=}"
    shift # past argument=value
    ;;
    --implementation=*)
    IMPLEMENTATION="${i#*=}"
    shift # past argument=value
    ;;
    --nestimators=*)
    NESTIMATORS="${i#*=}"
    shift # past argument=value
    ;;
    --nclasses=*)
    NCLASSES="${i#*=}"
    shift # past argument=value
    ;;
    --nfeatures=*)
    NFEATURES="${i#*=}"
    shift # past argument=value
    ;;
    --difficulty=*)
    DIFFICULTY="${i#*=}"
    shift # past argument=value
    ;;
    --nexamples=*)
    NEXAMPLES="${i#*=}"
    shift # past argument=value
    ;;
    --baseimplementation=*)
    BASE_IMPLEMENTATION="${i#*=}"
    shift # past argument=value
    ;;
    *)
    # unknown option
    ;;
esac
done

OUTPATH=$OUTPATH/$MODELNAME

mkdir -p $OUTPATH
if [ "$TYPE" = "cnn" ]; then
    python3 tests/data/generate_mnist.py --out $OUTPATH --float
else
    python3 tests/data/generate_data.py --out $OUTPATH --nclasses $NCLASSES --nfeatures $NFEATURES --difficulty $DIFFICULTY --nexamples $NEXAMPLES --float
fi

if [ "$TYPE" = "mlp" ] || [ "$TYPE" = "cnn" ]; then
    python3 tests/train_$TYPE.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME 
    ENDING="onnx"
else
    python3 tests/train_$TYPE.py --training $OUTPATH/training.csv --testing $OUTPATH/testing.csv --out $OUTPATH --name $MODELNAME  --nestimators $NESTIMATORS 
    ENDING="json"
fi

if [ -z "$BASE_IMPLEMENTATION" ]; then
    python3 fastinference/main.py --model $OUTPATH/$MODELNAME.$ENDING --feature_type "double" --out_path $OUTPATH --out_name "model" --implementation $IMPLEMENTATION 
else
    python3 fastinference/main.py --model $OUTPATH/$MODELNAME.$ENDING --feature_type "double" --out_path $OUTPATH --out_name "model" --implementation $IMPLEMENTATION --base_implementation $BASE_IMPLEMENTATION
fi

python3 ./tests/data/convert_data.py --file $OUTPATH/testing.csv --out $OUTPATH/testing.h --dtype "double" --ltype "unsigned int"
cp ./tests/main.cpp $OUTPATH
cp ./tests/CMakeLists.txt $OUTPATH
cd $OUTPATH
cmake . -DMODELNAME=$MODELNAME
make
./testCode
