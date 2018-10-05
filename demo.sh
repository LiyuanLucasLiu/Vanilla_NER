
# modify the following section
TRAIN_DATA="./data/ner/eng.train.iobes"
DEV_DATA="./data/ner/eng.testa.iobes"
TEST_DATA="./data/ner/eng.testb.iobes"

EMBEDDING_DATA="./embedding/glove.6B.100d.txt"

green=`tput setaf 2`
reset=`tput sgr0`

# mkdir
mkdir -p ./data

# pre-processing
echo ${green}=== Generating Dictionary ===${reset}
python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA

echo ${green}=== Encoding ===${reset}
python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA

# train & test
echo ${green}=== Training ===${reset}
python train_seq.py
