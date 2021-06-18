SOURCE_MODEL=resnet50_feature_noise
TARGET_MODEL="resnet50 densenet121 vgg16_bn resnet152 inception_v3"
BATCH_SIZE=20
SAMPLING_FREQ=1
MAX_ITER=300
LOG_INTERVAL=10
LR=4/255
EPSILON=16
LOSS=push-pull
VARIANT=mi-ti
DI=True
NOISE=uniform
FEATURE_NOISE_STD=0.035
SUBFOLDER=test

python3 main.py \
    --source-model $SOURCE_MODEL --target-model $TARGET_MODEL \
    --batch-size $BATCH_SIZE --sampling-frequency $SAMPLING_FREQ \
    --max-iterations $MAX_ITER --log-interval $LOG_INTERVAL \
    --lr $LR --epsilon $EPSILON --loss-fn $LOSS \
    --variant $VARIANT --di $DI \
    --input-noise $NOISE --feature-noise-std $FEATURE_NOISE_STD \
    --subfolder $SUBFOLDER