import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


from net import BCNet, NetType      # network impelementation
from data import BCData, DataSet    # data processing implementation

##-------------------------------------
# run()
# - main entry point for model training
#--------------------------------------
def run(_ds, _nt, _n_epoch, _data_fitgen=False, _savef='model.h5'):

    # 1. create BCData class object
    bcData = BCData(_ds, _fitgen=_data_fitgen)

    # 2. create Behavioral Cloning Network class object with bcData
    bcNet = BCNet(_bcdata=bcData, _netType=_nt)

    # 3. build network
    bcNet.buildNet()

    # 4. train
    bcNet.train(_n_epoch=_n_epoch)

    # 5. save
    bcNet.saveModel(_savef)

    # 6. plot result
    if _data_fitgen:
        bcNet.plot()


if __name__ == '__main__':

    ds = DataSet.ALL
    #ds = DataSet.PROVIDED
    nt = NetType.DNET
    #nt = NetType.LENET
    n_epoch = 50


    # LET'S RUN!
    run(ds, nt, n_epoch, _data_fitgen=False, _savef='model_dNet_all_Crop_Augmented.h5')
