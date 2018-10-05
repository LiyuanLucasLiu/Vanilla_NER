
# modify the following section
TRAIN_DATA="./data/ner/eng.train.iobes"
DEV_DATA="./data/ner/eng.dev.iobes"
TEST_DATA="./data/ner/eng.test.iobes"

EMBEDDING_DATA="./embedding/glove.6B.100d.txt"

# mkdir
mkdir -p ./data

# pre-processing
python pre_seq/gene_map.py --train_corpus $TRAIN_DATA --input_embedding $EMBEDDING_DATA
python pre_seq/encode_data.py --train_file $TRAIN_DATA --test_file $TEST_DATA --dev_file $DEV_DATA

# train & test
python train_seq.py
