CDVICE=3

NUM_LAYERS=3
BATCH_SIZE=10
NUM_CLASSES=27284
NUM_CLASSES_ACT=43

SCRIPT_FILE=/n/sd3/feng/ASR/exercise/trung/data/swda_full_vocab_old_split20_train.csv
SAVE_DIR=da_model

CUDA_VISIBLE_DEVICES=${CDVICE} python DA_model.py ${NUM_LAYERS} ${BATCH_SIZE} ${NUM_CLASSES} ${NUM_CLASSES_ACT} ${SCRIPT_FILE} ${SAVE_DIR}
