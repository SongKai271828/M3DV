from misc import set_gpu_usage

set_gpu_usage()

import densenet
import utils
import numpy as np
from Global import *

# 导入测试集时的参数
BATCH_SIZE_TEST = 3
DEAL_SIZE = 32


# 定义测试集类
test_set = utils.Dataset(
    data_path=test_path,
    label_path=test_label_path,
    batch=BATCH_SIZE_TEST,
    type='test',
    pre=True,
    deal_size=DEAL_SIZE,
    enhance=False,
)


model = densenet.get_compiled()

test_data, test_label = test_set.load_all()


def load_and_eval():
    results = np.zeros([117, ], dtype=np.float)
    for name in range(len(good_path)):
        model.load_weights(good_path[name])
        result = model.predict(x=test_data, batch_size=BATCH_SIZE_TEST, verbose=1)
        result1 = result[:, 1]
        # 综合推断结果
        results = results + result1 * lam[name]
    # 保存推断结果
    np.savetxt('result.csv', results, delimiter=',')


if __name__ == '__main__':
    load_and_eval()
