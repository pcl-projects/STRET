"""
Train using MirroredStrategy 
"""
#https://github.com/tensorflow/tensorflow/issues/36147

# the four aeguments are 100, 1000, -1, 2
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import numpy as np
import json

import pandas as pd
# Import TensorFlow
# import tensorflow_datasets as tfds
import tensorflow as tf
# import psutil
# import GPUtil as GPU
import random
import time
import tensorflow as tf

import threading

import psutil
import random

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import time
import threading
import pandas as pd

from tensorflow.keras.datasets import cifar10

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.models import Model

#from batchsizemanager import *

from worker2 import loss_global

# from tensorflow.keras.utils import np_utils
#from AlexNet_model import AlexNet

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3

MY_LIST = []
LOSSES_G = []
CHECK = False

# tf.compat.v1.enable_eager_execution()

loss_global = None
# loss_global = tf.constant([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]])
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 


# from tf.keras.models import Sequential

# from tf.keras.layers import Dense, Dropout
# from tf.keras.layers import Embedding
# from tf.keras.layers import LSTM
# tf.keras.backend.clear_session()

# tf.compat.v1.disable_eager_execution()
# tf.enable_eager_execution()
# tf.config.run_functions_eagerly(True)

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



def resnet_block(X, filter_size, filters, stride=1):

    # Save the input value for shortcut
    X_shortcut = X

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:

        X_shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    X = tf.keras.layers.Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second layer of the block
    X = tf.keras.layers.Conv2D(filters, kernel_size = filter_size, strides=(1, 1), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.add([X, X_shortcut])  # Add shortcut value to main path
    X = tf.keras.layers.Activation('relu')(X)

    return X




def resnet_model(input_shape, classes, name):

    # Define the input
    X_input = tf.keras.Input(input_shape)

    # Stage 1
    X = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Stage 2
    X = resnet_block(X, filter_size=3, filters=16, stride=1)
    X = resnet_block(X, filter_size=3, filters=16, stride=1)
    X = resnet_block(X, filter_size=3, filters=16, stride=1)
    X = resnet_block(X, filter_size=3, filters=16, stride=1)
    X = resnet_block(X, filter_size=3, filters=16, stride=1)

    # Stage 3
    X = resnet_block(X, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
    X = resnet_block(X, filter_size=3, filters=32, stride=1)
    X = resnet_block(X, filter_size=3, filters=32, stride=1)
    X = resnet_block(X, filter_size=3, filters=32, stride=1)
    X = resnet_block(X, filter_size=3, filters=32, stride=1)

    # Stage 4
    X = resnet_block(X, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
    X = resnet_block(X, filter_size=3, filters=64, stride=1)
    X = resnet_block(X, filter_size=3, filters=64, stride=1)
    X = resnet_block(X, filter_size=3, filters=64, stride=1)
    X = resnet_block(X, filter_size=3, filters=64, stride=1)

    # Average pooling and output layer
    X = tf.keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(classes, activation='softmax')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name=name)

    return model



def computation_time_thread(rpcClient):
    # global cpu_max 
    global loss_global 

    get_labels = loss_global.numpy()
    
    start = time.time()
    while(True):
        available_cpu = 2000 - psutil.cpu_percent(interval=None)
        pid = os.getpid()
        pid_use = psutil.Process(pid)
        memoryUse = pid_use.memory_info()[0]/2.**20
        available_memory = 32000 - memoryUse
        time.sleep(5)
        # return_from_rpc =rpcClient.update_batch_size(0, int(20),int(available_cpu), int(available_memory), 5, 128)
        # print(return_from_rpc)

        gradients_from_w2 = loss_global
        print("gradients from w2\n \n")
        print(gradients_from_w2)
        # if np.array_equal(get_labels, loss_global.numpy()) is not True:
        #     time_taken = time.time() -start
        #     print("time "+str(time_taken))
        #     get_labels = loss_global.numpy()
        #     start = time.time()

        # print(get_labels)
            

        # if StragglerManager.strag_found:
        #     print("straggler found")
        #     os._exit(1)

        # cpu_use=current_process.cpu_percent(interval=None)
        # memoryUse = current_process.memory_info()[0]/2.**20  # memory use in MB...

        # if cpu_use > cpu_max:
        #     cpu_max =cpu_use

        # if memoryUse > mem_max:
        #     mem_max = memoryUse
        # print(str(loss_global.numpy()))

        # get_labels = loss_global.numpy()
        # start = time.time()
        # while np.array_equal(get_labels, loss_global.numpy()) is True:
        #     end = time.time()
        #     get_labels = loss_global.numpy()
        #     time.sleep(1)
            

        # print("time: "+str(end-start))

        # time.sleep(1)
        # prev_labels = get_labels

        # print(loss_global.numpy())
        # tf.print(loss_global)
        
        # print(tf.get_static_value(loss_global))
        # time.sleep(1)
    # sess = tf.compat.v1.Session()
    # while True:
    #  sess.run(loss_global)
    #  time.sleep(1)

    # with tf.Session() as sess:
            # sess.run(init)




def main():
    set_tf_config()
    # train(20, 150, -1, 2)
    train()



def set_tf_config():

    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["localhost:40001", "localhost:40002", "localhost:40003", "localhost:40004"],#,"localhost:40005", "localhost:40006", "localhost:40007", "localhost:40008", "localhost:40009", "localhost:40010", "localhost:40011", "localhost:40012", "localhost:40013", "localhost:40014", "localhost:40015", "localhost:40016"],
            'ps': ["localhost:40000"],
        },
        'task': {'type': 'worker', 'index': 0},
    })

def train():
   

    # start = time.time()
    # pid = os.getpid()
    # py = psutil.Process(pid)
    # current_process = psutil.Process()
    # # current_process.cpu_affinity([0, 1, 2])
    # cpu=current_process.cpu_percent()


    # strategy = tf.distribute.experimental.ParameterServerStrategy()

    # communication_options = tf.distribute.experimental.CommunicationOptions(
    # implementation=tf.distribute.experimental.CommunicationImplementation.RING)
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication=tf.distribute.experimental.CollectiveCommunication.RING)

    # strategy = tf.distribute.MirroredStrategy()

    # batchSizeManager = BatchSizeManager(128, 2)
    # batchSizeManager = BatchSizeManager(FLAGS.batch_size, len(worker_hosts))

    # rpcServer = batchSizeManager.create_rpc_server(ps_hosts[0].split(':')[0])
    # rpcServer = batchSizeManager.create_rpc_server('localhost')
    # rpcServer.serve()
    # server.join()

    # time.sleep(2)
    # rpcClient = batchSizeManager.create_rpc_client('localhost')

    (X_train, y_train),(X_test,y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train/255.
    X_test = X_test/255.

    # val_labels = tf.one_hot(val_labels, 10, 1, 0)
    print(y_train.shape, y_test.shape)
    x_train = np.reshape(X_train, (-1, _HEIGHT, _WIDTH, _DEPTH))
    x_test = np.reshape(X_test, (-1, _HEIGHT, _WIDTH, _DEPTH))
    

    # y_train = tf.keras.utils.to_categorical(y_train)
    # y_test = tf.keras.utils.to_categorical(y_test)
    # print(y_train.shape, y_test.shape)

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    
    # x_batch_train, y_batch_train, new_batch = get_train_dataset(train_dataset, initial_batch_size)

    global MY_LIST
    global LOSSES_G
    global CHECK
    # global loss_global

    BUFFER_SIZE = len(x_train)

    BATCH_SIZE_PER_REPLICA = 512
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    EPOCHS = 1000


    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE) 

    # train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    # test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    my_dist_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE_PER_REPLICA)
    my_dist_test = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE_PER_REPLICA)

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(initial_batch_size)

    # val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # val_dataset = val_dataset.batch(initial_batch_size)

    with strategy.scope():
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)
          # reduction = tf.compat.v1.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


    with strategy.scope():
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    # model, optimizer, and checkpoint must be created under `strategy.scope`.
    with strategy.scope():
        # model = tf.saved_model.load('resnet_my_method/model_all_reduce')
        model=resnet_model(input_shape=(_HEIGHT, _WIDTH, _DEPTH), classes=10, name='ResNet32')

        # model = get_model((_HEIGHT, _WIDTH, _DEPTH), 10)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # optimizer =tf.keras.optimizers.SGD()
        # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    with strategy.scope():
        def train_step(inputs):
            # start = time.time()
            global loss_global

            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)

                loss_global = labels

            gradients = tape.gradient(loss, model.trainable_variables)
            # print(gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_accuracy.update_state(labels, predictions)

            # print(train_accuracy.result())

            # end = time.time()
            # print("computation time: "+str(end - start))

            return loss

        def test_step(inputs):
            images, labels = inputs

            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)

            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)



        def reduction(per_replica_losses):
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)


        from multiprocessing import util
        checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

        def _is_chief(task_type, task_id):
            return task_type is None or task_type == 'chief' or (task_type == 'worker' and
                                                               task_id == 0)

        def _get_temp_dir(dirpath, task_id):
            base_dirpath = 'workertemp_' + str(task_id)
            temp_dir = os.path.join(dirpath, base_dirpath)
            tf.io.gfile.makedirs(temp_dir)
            return temp_dir

        def write_filepath(filepath, task_type, task_id):
            dirpath = os.path.dirname(filepath)
            base = os.path.basename(filepath)
            if not _is_chief(task_type, task_id):
                dirpath = _get_temp_dir(dirpath, task_id)
            return os.path.join(dirpath, base)



        @tf.function
        def distributed_train_step(dataset_inputs):
            global CHECK
            global MY_LIST
            global LOSSES_G
            # per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            # start = time.time()

            print("starting train_step")
            per_replica_losses = strategy.experimental_run_v2(train_step,
                                                            args=(dataset_inputs,))

            # print(strategy.experimental_local_results(per_replica_losses))

            # end = time.time()
            # print("computation time: "+str(end - start))

            # return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
            #              axis=None)
            return per_replica_losses

        @tf.function
        def distributed_test_step(dataset_inputs):
            # return strategy.run(test_step, args=(dataset_inputs,))
            return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        global loss_global

        loss_global = tf.constant([[1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.]])

        # cthread = threading.Thread(target=computation_time_thread, args=(rpcClient,))
        # cthread.start()

        # task_type, task_id = (strategy.cluster_resolver.task_type,
        #               strategy.cluster_resolver.task_id)

        # checkpoint = tf.train.Checkpoint(model=model)

        # write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id)
        # checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

        # latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        # if latest_checkpoint:
        #     checkpoint.restore(latest_checkpoint)

        pid = os.getpid()
        pid_use = psutil.Process(pid)


        for epoch in range(EPOCHS):
            # my_dist_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE_PER_REPLICA)

            # my_dist_test = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE_PER_REPLICA)

            train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE) 

            train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
            test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


            total_loss = 0.0
            num_batches = 0
            # for x in my_dist_data:
            iterator = iter(test_dist_dataset)
            # iterator = iter(my_dist_test)

            for x in train_dist_dataset:
            # for x in my_dist_data:
                # if num_batches > 2:
                #     available_cpu = 2000 - psutil.cpu_percent(interval=None)
                #     pid = os.getpid()
                #     pid_use = psutil.Process(pid)
                #     memoryUse = pid_use.memory_info()[0]/2.**20
                #     available_memory = 32000 - memoryUse
                    
                    # return_from_rpc =rpcClient.update_batch_size(0, int(examples_per_sec),int(available_cpu), int(available_memory), 5, 128)
                    # print(return_from_rpc)


                if num_batches > 2:
                    file_read_cc_time = 'resnet_my_method/comp_time_all_reduce.csv'
                    df_cc_time = pd.read_csv(file_read_cc_time, header=None)
                    # print(df_itn)
                       
                    index_worker_all = pd.DataFrame(df_cc_time.values[:,0])
                    # print(itn_time_all)

                    computation_time_all = pd.DataFrame(df_cc_time.values[:,1])

                    communication_time_all = pd.DataFrame(df_cc_time.values[:,2])
                    # print(itn_time_all)

                    computation_time_all = computation_time_all.to_numpy()

                    communication_time_all =communication_time_all.to_numpy()

                    index_worker_all = index_worker_all.to_numpy()

                    worker_index_all = index_worker_all[len(index_worker_all)-2:len(index_worker_all)]

                    workers_computation_time = computation_time_all[len(computation_time_all)-2:len(computation_time_all)]

                    workers_communication_time = communication_time_all[len(communication_time_all)-2:len(communication_time_all)]
                    # print(workers_itn)

                    # print("workers_itn "+ str(workers_itn))

                    max_computation_time = max(workers_computation_time)
                    # print("max computation time "+str(max_computation_time[0]))

                    min_computation_time = min(workers_computation_time)
                    # print("min computation time "+str(min_computation_time[0]))


                    max_communication_time = max(workers_communication_time)
                    # print("max communication time "+str(max_communication_time[0]))

                    min_communication_time = min(workers_communication_time)
                    # print("min communication time "+str(min_communication_time[0]))


                    worker_index = 0
                    for each_w in workers_computation_time:
                        mult  = each_w[0] / min_computation_time[0]
                        # print(mult)
                        if mult > 1.5 and FLAGS.task_id == worker_index_all[worker_index][0]:
                            computation_straggler = True

                        worker_index = worker_index + 1

                    worker_index = 0
                    for each_w in workers_communication_time:
                        mult  = each_w[0] / min_communication_time[0]
                        # print(mult)
                        if mult > 1.5 and FLAGS.task_id == worker_index_all[worker_index][0]:
                            communication_straggler = True

                        worker_index = worker_index + 1


                if computation_straggler and communication_straggler:
                    both_straggler = True


                start = time.time()
                # total_loss += distributed_train_step(x)

                per_replica_losses = distributed_train_step(x)
                computation_time = time.time() -start
                # print("computation time: "+str(computation_time))

                start_c = time.time()
                total_loss += reduction(per_replica_losses)
                try:
                    total_loss += reduction(loss_global)
                except Exception as  e: 
                    print("")

                communication_time = time.time() - start_c
                # print("communication_time: "+str(communication_time))
                examples_per_sec = BATCH_SIZE_PER_REPLICA/computation_time

                num_batches += 1

                # checkpoint_manager.save()
                # if not _is_chief(task_type, task_id):
                #     tf.io.gfile.rmtree(write_checkpoint_dir)

                # checkpoint.save('resnet_my_method/model_ar')
                
                # data_comp_time = []
                # data_comp_time.append(0)
                # data_comp_time.append(computation_time)
                # data_comp_time.append(communication_time)

                # data_comp_time_csv = []
                # data_comp_time_csv.append(data_comp_time)


                # import csv
                # file_comp_time = 'resnet_my_method/comp_time_all_reduce.csv'
                # with open(file_comp_time, 'a', newline ='') as file:
                #     wr_comp_time = csv.writer(file)
                #     wr_comp_time.writerows(data_comp_time_csv)


                # print(" train accuracy: "+str(train_accuracy.result()*100))
                st_train = str(train_accuracy.result())
                
                split_1= st_train.split("(")

                split_2 = split_1[1].split(", ")

                print("train accuracy: "+str(split_2[0]))

                try:
                    distributed_test_step(next(iterator))
                except Exception as  e:  
                    iterator = iter(test_dist_dataset)
                    # iterator = iter(my_dist_test)
                    # print("exception")

                print("calling distributed_test_step")
                # print(test_dist_dataset[0])
                # print(" test accuracy: "+str(test_accuracy.result()*100))

                st_test = str(test_accuracy.result())
                
                split_1_1= st_test.split("(")

                split_2_2 = split_1_1[1].split(", ")

                print("test accuracy: "+str(split_2_2[0]))


                print("completed step: " +str(num_batches))

                data = []
                data.append(1)
                data.append(computation_time + communication_time)
                data.append(examples_per_sec)
                # data.append(train_accuracy.result())
                # data.append(test_accuracy.result())
                data.append(split_2[0])
                data.append(split_2_2[0])

                data_csv = []
                data_csv.append(data)

                import csv
                file_data = 'my_allreduce_resnet/my_resnet_data1gpu.csv'
                with open(file_data, 'a', newline ='') as file:
                    wr2 = csv.writer(file)
                    wr2.writerows(data_csv)

                c_data = []
                c_data.append(1)
                c_data.append(computation_time + communication_time)
                c_data.append(examples_per_sec)
                # data.append(train_accuracy.result())
                # data.append(test_accuracy.result())
                c_data.append(split_2[0])
                c_data.append(split_2_2[0])

                c_data_csv = []
                c_data_csv.append(c_data)

                import csv
                c_file_data = 'my_allreduce_resnet/my_resnet_datagpu.csv'
                with open(c_file_data, 'a', newline ='') as file:
                    wr2_c = csv.writer(file)
                    wr2_c.writerows(c_data_csv)



            train_loss = total_loss / num_batches

            # TEST LOOP
            # for x in my_dist_test:
            for x in test_dist_dataset:
                distributed_test_step(x)


            # if epoch % 2 == 0:
            #     checkpoint.save(checkpoint_prefix)

            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
              "Test Accuracy: {}")
            print (template.format(epoch+1, train_loss,
                         train_accuracy.result()*100, test_loss.result(),
                         test_accuracy.result()*100))

            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

            # my_dist_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE_PER_REPLICA)

            # my_dist_test = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE_PER_REPLICA)
        # initial_batch_size = 128
        # n_epoch = 5
        # model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
        # model.compile(loss='mse', optimizer='sgd')
        # model = get_model()

        
        # model =  get_model((_HEIGHT, _WIDTH, _DEPTH), 10) #AlexNet(x = inputs, keep_prob = 0.5, num_classes = 10)

        # model=resnet_model(input_shape=(_HEIGHT, _WIDTH, _DEPTH), classes=classes, name='ResNet32')

        # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        # optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # model.compile(loss= 'categorical_crossentropy' , optimizer='sgd', metrics=[ 'accuracy' ])
        # print("Model Summary of ",model_type)
        # print(model.summary())



        
        print("model ok")
        
        # train_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(batch_size).shuffle(256)

        print("train dataset ok")

        filename='log_worker.csv'
        history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

        
        time_callback = TimeHistory()



        # model.fit(X_train, y_train,
        #   batch_size=batch_size,
        #   epochs=300,
        #   validation_data=(X_test, y_test),
        #   shuffle=True,
        #   callbacks = [history_logger, time_callback])

        # times = time_callback.times
        # print(times)

        # data =[]

        # for i in range(len(times)):
        #     data.append(times[i])
        # name = 'test1_w1'
        # data.append(times)
        # data.append(step)
        # data.append(gs)
        # data.append(loss_value)
        # data.append(examples_per_sec)
        # data.append(sec_per_batch)
        # data.append(duration)
        # data.append(cpu_use)
        # data.append(memoryUse)
        # data.append(cpu_max)
        # data.append(mem_max)
        # data.append(net_usage/duration)
        # data.append(accuracy)
        # data.append(val_accuracy)
        
        # data_csv = []
        # data_csv.append(data)
        # import csv
        # file_nam = 'ps_' + '0' + '.csv' 
        # with open(file_nam, 'a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerows(data_csv)
        
        # test_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(batch_size).shuffle(256)

        # model.fit(train_dataset, epochs = n_epoch)
        # # cpu=current_process.cpu_percent()
        # # memoryUse = py.memory_info()[0]/2.**20  # memory use in MB...
        # # io_counters = current_process.io_counters() 

        # # dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(10).shuffle(1000)
        # # model.fit(dataset, epochs=20)

        # model.evaluate(test_dataset)
      
def compare_grad(prev_grad, current_grad):
    # diff_ten = tf.sub(prev_grad, grad)
    difference = []
   # initialization of result list
    global CHECK
    zip_object = zip(prev_grad, current_grad)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)
        # append each difference to list

    # if np.all(abs(difference[:]) >= 0.01):
    #     print("True")
    # # print(difference)

    # for diff in difference:
    #     for element in diff:
    # #         if abs(ele)
    # for diff in difference:
    #     if all(abs(element) >= 0.001 for element in diff):
    #         print(True)

    grad_diff = np.array(difference)
    grad_diff = grad_diff.flatten()

    for element in grad_diff:
        element_flat = element.flatten()
        # kk =all(abs(ele) >= 0.001 for ele in element)
        # print(kk)

        for ele in element_flat:
            if (abs(ele) >= 0.0000001):
                CHECK = True
                break
                # print("hhh")

        # print(element)

    # print(grad_diff.shape)



def get_model(input_shape, num_classes):

    # inputs = tf.keras.Input(shape=input_shape, name="digits")
    # x1 = tf.keras.layers.Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=input_shape, activation='relu')(inputs)
    # x2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x1)
    # x3 = tf.keras.layers.Conv2D(256,(5,5),padding='same',activation='relu')(x2)
    # x4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x3)
    # x5 = tf.keras.layers.Conv2D(384,(3,3),padding='same',activation='relu')(x4)
    # x6 = tf.keras.layers.Conv2D(384,(3,3),padding='same',activation='relu')(x5)
    # x7 = tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu')(x6)
    # x8 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x7)
    # x9 =  tf.keras.layers.Flatten()(x8)
    # x10 = tf.keras.layers.Dense(4096, activation='relu')(x9)
    # x11 = tf.keras.layers.Dropout(0.4)(x10)
    # x12 = tf.keras.layers.Dense(4096, activation='relu')(x11)
    # x13 = tf.keras.layers.Dropout(0.4)(x12)
    # outputs = tf.keras.layers.Dense(num_classes,activation='softmax')(x13)

    # model = Model(inputs=inputs, outputs=outputs)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96,kernel_size=(3,3),strides=(4,4),input_shape=input_shape, activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(256,(5,5),padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    model.add(tf.keras.layers.Conv2D(384,(3,3),padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv2D(384,(3,3),padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))


    # inputs = keras.Input(shape=(784,), name="digits")
    # x1 = layers.Dense(64, activation="relu")(inputs)
    # x2 = layers.Dense(64, activation="relu")(x1)
    # outputs = layers.Dense(10, name="predictions")(x2)
    # model = keras.Model(inputs=inputs, outputs=outputs)

    return model

  #model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

  #model.summary()



if __name__ == '__main__':
    main()
