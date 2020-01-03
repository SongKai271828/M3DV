# sampleSubmission.csv所在路经
test_label_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/sampleSubmission.csv'
# 测试集数据所在路径
test_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/data_test/'

# train_val.csv所在路径
train_label_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/train_val.csv'
# 训练集数据所在路径
train_path = '/media/songkai/E4824F12824EE91E/3d_voxel/data/data_train/'
# 模型保存路径
save_path = '/home/songkai/PycharmProjects/voxels/keras_models/-{epoch:02d}.h5'

# 已保存模型
good_path = [
    '-694.h5',
    '-685.h5',
]

# 多个模型推断时每个模型的权重
lam = [0.8, 0.2]
