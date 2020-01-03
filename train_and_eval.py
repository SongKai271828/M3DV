from misc import set_gpu_usage

set_gpu_usage()

import densenet

from keras.callbacks import ModelCheckpoint
import utils
import numpy as np
import keras
from Global import *


BATCH_SIZE = 6		    # 训练集batch size
BATCH_SIZE_TEST = 3	    # 测试集batch size
DEAL_SIZE = 32		    # 截取数据尺寸
EPOCH = 30		        # 训练代数
TRAIN_SIZE = 1020	    # 训练集大小


# 训练集类定义
train_set = utils.Dataset(
    data_path=train_path,
    label_path=train_label_path,
    batch=BATCH_SIZE,
    type='train',
    pre=True,
    deal_size=DEAL_SIZE,
    enhance=True,
    intep=False
)

# 测试集类定义
test_set = utils.Dataset(
    data_path=test_path,
    label_path=test_label_path,
    batch=BATCH_SIZE_TEST,
    type='test',
    pre=True,
    deal_size=DEAL_SIZE,
    enhance=False,
    intep=False
)


# 生成模型
model = densenet.get_compiled()


# 导入全部训练数据
all_data, all_label = train_set.load_all()


# 打乱训练集顺序
for _ in range(4):
    state = np.random.get_state()
    np.random.shuffle(all_data)
    np.random.set_state(state)
    np.random.shuffle(all_label)


# 从训练集中分出一部分作为验证集
train_data = all_data[0:TRAIN_SIZE]
train_label = all_label[0:TRAIN_SIZE]
vali_data = all_data[TRAIN_SIZE:]
vali_label = all_label[TRAIN_SIZE:]


# 导入全部测试集数据
test_data, test_label = test_set.load_all()


def train_and_eval():
    checkpointer = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=False, save_weights_only=True)
    model.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=2, shuffle=True,
              validation_data=(vali_data, vali_label), callbacks=[checkpointer])
    # result = model.predict(x=test_data, batch_size=BATCH_SIZE_TEST, verbose=1)
    # result = result[:, 1]
    # np.savetxt('result_keras' + '.csv', result, delimiter=',')
    # model.save(save_path+'checkpoint_keras.h5')


if __name__ == '__main__':
    train_and_eval()
