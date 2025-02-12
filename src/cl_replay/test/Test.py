import tensorflow   as tf
import numpy        as np
import argparse, os

from cl_replay.architecture.ar.model        import DCGMM
from cl_replay.architecture.ar.layer        import GMM_Layer, Folding_Layer, Readout_Layer, MFA_Layer
from cl_replay.architecture.ar.callback     import Log_Protos, Set_Model_Params, Early_Stop
from cl_replay.architecture.ar.adaptor      import AR_Supervised
from cl_replay.architecture.ar.generator    import AR_Generator

from cl_replay.api.layer.keras  import Input_Layer, Reshape_Layer, Concatenate_Layer
from cl_replay.api.callback     import Log_Metrics
from cl_replay.api.data         import Dataset, Sampler
from cl_replay.api.utils        import change_loglevel


def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print ("MINMAX", x_train.min(), x_train.max())

    #x_train    = x_train.reshape(60000, 1, 1, 784).astype("float32") / 255.0
    #x_test     = x_test.reshape(10000, 1, 1, 784).astype("float32") / 255.0

    x_train     = (x_train.reshape(60000, 28, 28, 1).astype("float32")) / 255.0
    x_test      = (x_test.reshape(10000, 28, 28, 1).astype("float32")) / 255.0

    print ("MINMAX", x_train.min(), x_train.max())

    # Split for train/val4
    x_val       = x_train[-10000:]
    y_val       = y_train[-10000:]

    x_train     = x_train[:50000]
    y_train     = y_train[:50000]

    onehot_y_train = np.zeros((y_train.shape[0], y_train.max() + 1), dtype=np.float32)
    onehot_y_train[ np.arange(y_train.shape[0]), y_train ] = 1

    onehot_y_test = np.zeros((y_test.shape[0], y_test.max() + 1), dtype=np.float32)
    onehot_y_test[ np.arange(y_test.shape[0]), y_test ] = 1

    #train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    return x_train, onehot_y_train, x_test, onehot_y_test


def return_functional_model(choice, input_shape, gmm_k, vis_path, sbs):
    inputs      = Input_Layer(              layer_name="L0_INPUT", prefix="L0_", shape=input_shape).create_obj()

    if choice == "simple":
        fold_1  = Folding_Layer(            layer_name="L1_FOLD", L1_patch_height=input_shape[0], L1_patch_width=input_shape[1], L1_stride_x=1, L1_stride_y=1, sampling_batch_size=sbs)(inputs)
        gmm_1   = GMM_Layer(                layer_name="L2_GMM", L2_K=gmm_k, L2_lambda_sigma=0.1, L2_lambda_mu=1., L2_lambda_pi=0., L2_eps0=0.011, L2_epsInf=0.01, L2_sampling_I=-1, L2_sampling_divisor=10., sampling_batch_size=sbs)(fold_1)
        outputs = gmm_1

    elif choice == "simple_cl":
        fold_1  = Folding_Layer(            layer_name="L1_FOLD", prefix="L1_", input_layer=0, L1_patch_width=input_shape[0], L1_patch_height=input_shape[1], L1_stride_x=1, L1_stride_y=1, sampling_batch_size=sbs,)(inputs)
        gmm_1   = GMM_Layer(                layer_name="L2_GMM", prefix="L2_", input_layer=1, L2_K=gmm_k, L2_lambda_sigma=0.1, L2_lambda_mu=1., L2_lambda_pi=0., L2_eps0=0.011, L2_epsInf=0.01, L2_sampling_I=-1, L2_sampling_divisor=10., sampling_batch_size=FLAGS.batch_size)(fold_1)
        cl      = Readout_Layer(            layer_name="L3_READOUT", prefix="L3_", input_layer=2, L3_num_classes=10, L3_loss_function ="mean_squared_error", sampling_batch_size=sbs)(gmm_1)
        outputs = cl

    elif choice == "simple_mfa":
        fold_1  = Folding_Layer(            layer_name="L1_FOLD", L1_patch_width=28, L1_patch_height=28, L1_stride_x=2, L1_stride_y=2)(inputs)
        gmm_1   = MFA_Layer(                layer_name="L2_GMM", L2_K=gmm_k, L2_l=4, L2_lambda_E=0.0, L2_lambda_gamma=0.1)(fold_1)
        cl      = Readout_Layer(            layer_name="L3_READOUT", L3_num_classes=10, L3_loss_function ="softmax_cross_entropy")(gmm_1)
        outputs = cl

    elif choice == "simple2":
        fold_1  = Folding_Layer(            layer_name="L1_FOLD", L1_patch_width=8, L1_patch_height=8, L1_stride_x=2, L1_stride_y=2)(inputs)
        gmm_1   = GMM_Layer(                layer_name="L2_GMM", L2_K=gmm_k)(fold_1)
        outputs = gmm_1

    elif choice == "2gmm":
        fold_1  = Folding_Layer(            layer_name="L1_FOLD", L1_patch_width=8, L1_patch_height=8, L1_stride_x=2, L1_stride_y=2)(inputs)
        gmm1l   = GMM_Layer(                layer_name="L2_GMM", L2_K=gmm_k, L2_wait=-1, L2_sampling_divisor=10., L2_eps0=0.0011, L2_epsInf=0.001, L2_sampling_S=1, L2_sampling_I=-1 )
        gmm_1   = gmm1l(fold_1)
        fold_2  = Folding_Layer(            layer_name="L3_FOLD", L3_patch_width=11, L3_patch_height=11, L3_stride_x=1, L3_stride_y=1)(gmm_1)
        gmm_2   = GMM_Layer(                layer_name="L4_GMM", L4_K=gmm_k, L4_wait=-1, L4_sampling_divisor=10., L4_sampling_I=-1, L2_eps0=0.0011, L2_epsInf=0.001)(fold_2)
        outputs = gmm_2

    elif choice == "2gmm_class":
        fold_1  = Folding_Layer(            layer_name="L1_FOLD", L1_patch_width=8, L1_patch_height=8, L1_stride_x=2, L1_stride_y=2)(inputs)
        gmm_1   = GMM_Layer(                layer_name="L2_GMM", L2_K=gmm_k, L2_eps0=0.011, L2_epsInf=0.01, L2_lambda_sigma=0.1, L2_sampling_divisor=10)(fold_1)
        fold_2  = Folding_Layer(            layer_name="L3_FOLD", L3_patch_width=11, L3_patch_height=11, L3_stride_x=1, L3_stride_y=1)(gmm_1)
        gmm_2   = GMM_Layer(                layer_name="L4_GMM", L4_K=gmm_k, L4_eps0=0.011, L4_epsInf=0.01, L4_lambda_sigma=0.1, L4_sampling_divisor=10)(fold_2)
        rs_1    = Reshape_Layer(            layer_name="L5_RESHAPE", target_shape=[1,1,11*11*gmm_k])(gmm_1)
        rs_2    = Reshape_Layer(            layer_name="L6_RESHAPE", target_shape=[1,1,gmm_k])(gmm_2)
        concat  = Concatenate_Layer(        layer_name="L7_CONCAT")([rs_1, rs_2])
        cl      = Readout_Layer(            layer_name="L8_READOUT", L8_num_classes = 10, L8_loss_function ="softmax_cross_entropy")(concat)
        outputs = cl

    return DCGMM(                           inputs=inputs, outputs=outputs, name="test", vis_path=vis_path,)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",           type=str, default="variants", choices=["train", "variants", "sample"])
    parser.add_argument("--model",          type=str, default="simple")
    parser.add_argument("--epochs",         type=int, default=4)
    parser.add_argument("--batch_size",     type=int, default=50)
    parser.add_argument("--input_shape",    nargs='+', type=int, default=[48, 48, 3])
    parser.add_argument("--gmm_k",          type=int, default=25)
    parser.add_argument("--sampling_I",     type=int, default=-1)
    parser.add_argument("--dataset_dir",    type=str, default="/home/ak/datasets")
    parser.add_argument("--dataset_name",   type=str, default="fruits.npz")
    parser.add_argument("--dataset_load",   type=str, default="from_npz")
    parser.add_argument("--vis_path",       type=str, default="/home/ak/exp-results/")
    parser.add_argument("--log_path",       type=str, default="/home/ak/exp-results/")
    parser.add_argument("--ckpt_dir",       type=str, default="/home/ak/exp-results/")
    
    

    FLAGS = parser.parse_args()
    exp_id = 'test.py'

    """ 
    Run this file via:   
        export PYTHONPATH=$PYTHONPATH:/path/to/scclv2/src/
        cd /path/to/scclv2/src/
        python3 ./cl_replay/test/Test.py
    """

    #-------------------------------------------- DATA PREP
    # x_train, y_train, x_test, y_test = load_mnist()
    # samples_train = x_train.shape[0]

    batch_size = FLAGS.batch_size
    test_batch_size = batch_size
    epochs = FLAGS.epochs

    ds = Dataset(
        epochs=epochs,
        dataset_dir=FLAGS.dataset_dir,
        dataset_name=FLAGS.dataset_name,
        dataset_load=FLAGS.dataset_load,
        vis_path=FLAGS.vis_path,
        log_path=FLAGS.log_path,
        ckpt_dir=FLAGS.ckpt_dir,
        renormalizeC=1.,
        np_shuffle="no",
    )
    ds_info = ds.properties
    h,w,c,num_classes = ds_info['dimensions'][0], ds_info['dimensions'][1], ds_info['num_of_channels'], ds_info['num_classes']

    generated_dataset  = None  # saves ref to generator data
    initial_classes = [0, 1, 2]
    ds_train, ds_test, samples_train, samples_test = ds.get_dataset(task_data=initial_classes)

    cur_xs, cur_ys = ds_train
    t1_test_xs, t1_test_ys = ds_test

    train_iterations = samples_train // batch_size
    test_iterations = samples_test // test_batch_size

    #-------------------------------------------- MODEL CREATION
    model = return_functional_model(FLAGS.model, FLAGS.input_shape, FLAGS.gmm_k, FLAGS.vis_path, FLAGS.batch_size)
    model.summary()
    model.compile(run_eagerly=True)
    change_loglevel("DEBUG")

    l_p = Log_Protos(exp_id=exp_id, save_when="epoch_end", log_connect="no", vis_path=FLAGS.vis_path)
    l_m = Log_Metrics(
        exp_id=exp_id, vis_path=FLAGS.vis_path, log_path=FLAGS.log_path,
        log_training="no", log_connect="no", save_when="train_end"
    )
    s_m_p = Set_Model_Params()
    e_s = Early_Stop(patience=64, ro_patience=False)

    train_callbacks = [l_p, l_m, s_m_p, e_s]
    eval_callbacks = [l_p, l_m]
    
    #-------------------------------------------- REPLAY STRUCTS
    sampler = Sampler(
        batch_size=batch_size
    )
    adaptor = AR_Supervised(
        sampling_batch_size=FLAGS.batch_size,
        samples_to_generate=1., # mul. factor
        sample_variants="yes"
    )
    generator = AR_Generator(
        model = model,
        data_dims = (h,w,c,num_classes)
    )
    # add data to sampler and set sampling proportions, for now only 1 data source added
    sampler.add_subtask(xs=cur_xs, ys=cur_ys)
    sampler.set_proportions([1.])
    # prepare adaptor (set model, generator, input dimensions)
    adaptor.set_model(model)
    adaptor.set_input_dims(h, w, c, num_classes)
    adaptor.set_generator(generator)

    #-------------------------------------------- TRAIN/EVAL PIPELINE
    if FLAGS.mode == "train":
        task = 1 
        adaptor.before_subtask(task)     
        history = model.fit(
            sampler(),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=train_iterations,
            callbacks=train_callbacks,
            verbose=2
        )
        adaptor.after_subtask()

        model.test_task = 'T1'
        tmp = model.evaluate(t1_test_xs, t1_test_ys,
            steps=test_iterations, 
            callbacks=eval_callbacks
        )

        model.save_weights(filepath=f'./checkpoints/{model.name}', overwrite=True)

    elif FLAGS.mode == "sample":
        model.load_weights(filepath=f'./checkpoints/{model.name}')

        tmp             = model.evaluate(x_test, y_test)
        samples         = model.sample_one_batch()
        vars            = model.do_variant_generation(x_train[0:100], selection_layer_index=-1)

        model.save_npy_samples(samples,                     prefix="results/sampling_")
        model.save_npy_samples(tf.constant(x_train[0:100]), prefix="results/originals_")
        model.save_npy_samples(vars,                        prefix="results/variants_")
    
    #-------------------------------------------- VARIANT GEN.
    elif FLAGS.mode == 'variants':
        #----------- task = 1 --> initial training (no replay)
        task = 1        
        #------ TRAIN ROUTINE
        adaptor.before_subtask(task)
        history = model.fit(
            sampler(),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=train_iterations,
            callbacks=train_callbacks,
            verbose=2
        )
        adaptor.after_subtask()

        #------ TEST ROUTINE
        model.test_task = 'T1'
        tmp = model.evaluate(t1_test_xs, t1_test_ys,
            steps=test_iterations, 
            callbacks=eval_callbacks
        )

        #------ SAVE/LOAD: Clone the current model and set the generator to point towards a copy of the "old" model instance!
        # SAVE MODEL WEIGHTS TO CHECKPOINT FILE
        ckpt_dir = './checkpoints/'
        model_type = 'DCGMM'
        chkpt_filename = os.path.join(
            ckpt_dir,
            f'{exp_id}-{model_type.lower()}-clone.ckpt')
        model.save_weights(chkpt_filename)
    
        # CREATE/COMPILE A CLONE
        cloned_model = return_functional_model(FLAGS.model, FLAGS.input_shape, FLAGS.gmm_k, FLAGS.vis_path, FLAGS.batch_size)
        cloned_model.summary()
        cloned_model.compile(run_eagerly=True)
        change_loglevel("DEBUG")

        #print(tf.train.list_variables(chkpt_filename))
        try: # LOAD WEIGHTS INTO THE CLONE!
            cloned_model.load_weights(chkpt_filename)   
            print(f'RESTORED MODEL FROM CHECKPOINT FILE "{chkpt_filename}"...')
        except Exception as ex:
            import traceback
            print(traceback.format_exc())
            print(f'A PROBLEM WAS ENCOUNTERED LOADING THE MODEL FROM CHECKPOINT FILE "{chkpt_filename}": {ex}')
            raise ex
        for a, b in zip(model.weights, cloned_model.weights):
            np.testing.assert_allclose(a.numpy(), b.numpy())

        # GENERATOR NOW HOLDS THE REFERENCE TO CLONED MODEL (STATIC WEIGHTS)!
        generator.set_model(cloned_model)
        #------ GENERATE
        #----------- task >= 1 --> replay training (var. gen.)
        task += 1
        #----------- load new data for next task, current xs data is used to gen. variants!
        new_classes = [3, 4, 5]
        ds_train, ds_test, samples_train, samples_test = ds.get_dataset(task_data=new_classes)     
        
        cur_xs, cur_ys = ds_train
        t2_test_xs, t2_test_ys = ds_test
        
        train_iterations = samples_train // batch_size
        test_iterations = samples_test // test_batch_size
        train_iterations *= 2   # replaying 50, 50 ... so simply double task_iters

        #vars = model.do_variant_generation(cur_xs, selection_layer_index=-1)
        gen_xs, gen_ys = adaptor.generate(task=task, xs=cur_xs, generate_labels=False)
        print(gen_xs.shape)
        
        sampler.reset()

        sampler.add_subtask(xs=gen_xs, ys=gen_ys)     # SUBTASK 0 -> generated data
        sampler.add_subtask(xs=cur_xs, ys=cur_ys)     # SUBTASK 1 -> current task real data

        sampler.set_proportions([50., 50.])

        #------ TRAIN ROUTINE
        adaptor.before_subtask(task)
        history = model.fit(
            sampler(),
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=train_iterations,
            callbacks=train_callbacks,
            verbose=2
        )
        adaptor.after_subtask()

        #------ TEST ROUTINE
        model.test_task = 'T1'
        tmp = model.evaluate(t1_test_xs, t1_test_ys,
            steps=test_iterations,
            callbacks=eval_callbacks
        )
        model.test_task = 'T2'
        tmp = model.evaluate(t2_test_xs, t2_test_ys,
            steps=test_iterations,
            callbacks=eval_callbacks
        )
