# SCalable Continual Learning (SCCL)

## General information
A machine learning (ML) framework written to facilitate the experimental evaluation of various continual learning (CL) strategies, primarily using TensorFlow, Keras and NumPy. This is also the code repository for the scientific papers **"Adiabatic replay for continual learning"**, **"An analysis of best-practice strategies for replay and rehearsal in continual learning"**, and **"Continual Unlearning through Memory Suppression"**. We provide detailed instructions on how to reconstruct the experiments from these articles.

**"Adiabatic replay for continual learning" (IJCNN 2024)** \
The paper introduces a generative replay-based approach to CL, termed adiabatic replay (AR), which achieves CL at constant time and memory complexity by exploiting the (very common) situation where each new learning phase is adiabatic, i.e. represents only a small addition to existing knowledge. AR is evaluated on some common task splits for MNIST, Fashion-MNIST, E-MNIST, and a feature-encoded version of SVHN and CIFAR-10.

**"An analysis of best-practice strategies for replay and rehearsal in continual learning" (CVPR - CLVISION WORKSHOP 2024)** \
This study evaluates the impact of various design choices in the context of class-incremental continual learning using replay methods. The investigation focuses on experience replay, generative replay using either VAEs or GANs and more general replay strategies such as loss weighting. It is evaluated on a variety of task splits for MNIST, Fashion-MNIST, E-MNIST, and a latent version of SVHN and CIFAR-10.

**"Continual Unlearning through Memory Suppression" (ESANN 2025)** \
We uncover surprisingly effective synergies between the field of continual learning (CL) and machine unlearning (MUL). It builds on the scenario of class-incremental learning (CIL), but incorporates suppression requests to demand the deletion of already consolidated class-level data. A trivial but effective strategy called "Replay to Suppress" (RTS) is used to achieve class-level forgetting by exploiting the inevitable effect of CF when replaying past data statistics. The evaluation was performed for AR, DGR, ER, Selective Amnesia on the MNIST and Fashion-MNIST benchmarks, as well as the latent encoded versions of SVHN and CIFAR-10.

## Cite our work
If you (re-)use the code in this repository for your own research, feel free to cite one or more of our published works:
```
@inproceedings{krawczyk2024adiabatic,
  title={Adiabatic replay for continual learning},
  author={Krawczyk, Alexander and Gepperth, Alexander},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```
```
@inproceedings{krawczyk2024analysis,
  title={An Analysis of Best-practice Strategies for Replay and Rehearsal in Continual Learning},
  author={Krawczyk, Alexander and Gepperth, Alexander},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4196--4204},
  year={2024}
}
```
```
@TBA{krawczyk2025rts,
}
```
**The full-text articles can be found online:**
1) ["Adiabatic replay for continual learning"](https://ieeexplore.ieee.org/document/10651381)
2) ["An Analysis of Best-practice Strategies for Replay and Rehearsal in Continual Learning"](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Krawczyk_An_Analysis_of_Best-practice_Strategies_for_Replay_and_Rehearsal_in_CVPRW_2024_paper.pdf)
3) ["Continual Unlearning through Memory Suppression"](TBA)

## Note
**Dependencies:**
- The code is tested and working for Python 3.10+ and the python packages listed in `requirements.txt`.
- All required packages can be installed via `pip install -r requirements.txt`, consider using a virtual environment or a containerized solution for the install.
- GPU-support for NVIDIA graphics cards with CUDA and CUDNN, see the current [compatibility chart](https://www.tensorflow.org/install/source#gpu).

**Bugs / Unexpected behaviour:**
If you encounter any bugs, unexpected behavior, have open questions or simply need support with this source code - feel free to contact us via mail: _Alexander.Krawczyk@cs.hs-fulda_, _Alexander.Gepperth@cs.hs-fulda_

## Using the API
### Running pre-defined experiments:
- Each bash file located in the directoory `./bash/*` represents a specific experiment, defining the data, model architecture, layers, hyperparameters, sampling procedure, data generation process, and so on.
- Once a file is invoked, it runs the experiment in an automated way:
   1) Before running the files, make sure that all the necessary dependencies are installed on the system side and that the environment variables are set correctly.
   2) (Opt.) Manipulate the experiment, model and layer parameters as required.
   3) Run the bash file/experiment as a shell command, e.g.,
   `bash /PATH/TO/scclv2/bash/<SOME_EXPERIMENT>.bash`. The results will be automatically saved to the file system after completion.

### Reconstructing experiments for AR and RTS:
To reconstruct the experiments presented in our articles, please run the experiments from the `./bash/examples` directory. Each of these files represents at least one benchmark for each model evaluated. Things like the order of tasks and the dataset have to be adjusted accordingly.

### Generating latent-encoded datasets for SVHN and CIFAR-10:
To create and use the latent-encoded versions of SVHN and CIFAR-10 as shown in our experiments, please run the two commands below first. This will run the contrastive training of the extractor network and create the numpy archive file for each dataset.

**Generate (latent) SVHN:**
```
export PROJ_PATH="$HOME/PATH/TO/scclv2"
python3 "${PROJ_PATH}/src/cl_replay/api/data/encoding/DatasetEncoder.py \
    --encode_ds svhn_cropped \
    --encode_split train \
    --split 10 \
    --out_dir $HOME/custom_datasets/encoded \
    --out_name svhn-7-ex \
    --architecture resnet_v2.ResNet50V2 \
    --include_top no \
    --bootstrap no \
    --pooling avg \
    --pretrain yes \
    --pretrain_ds svhn_cropped \
    --pretrain_split extra[:7%] \
    --pretrain_epochs 100 \
    --batch_size 1024 \
    --output_layer post_relu \
    --augment_data yes \
    --contrastive_learning yes \
    --contrastive_method simclr
```
**Generate (latent) CIFAR-10:**
```
export PROJ_PATH="$HOME/PATH/TO/scclv2"
python3 "${PROJ_PATH}/src/cl_replay/api/data/encoding/DatasetEncoder.py \
    --encode_ds cifar10 \
    --encode_split train[50%:] \
    --split 10 \
    --out_dir $HOME/custom_datasets/encoded \
    --out_name cifar10-50-ex \
    --architecture resnet_v2.ResNet50V2 \
    --include_top no \
    --bootstrap no \
    --pooling avg \
    --pretrain yes \
    --pretrain_ds cifar10 \
    --pretrain_split train[:50%] \
    --pretrain_epochs 100 \
    --batch_size 1024 \
    --output_layer post_relu \
    --augment_data yes \
    --contrastive_learning yes \
    --contrastive_method simclr
```

### Some settings to consider:
You can adapt the experiments to your own needs by manipulating the bash files or create your own. Feel free to experiment with different parameters. Some examples of how to adjust the parameters are shown below:
1) **Data settings:** \
    ```--dataset_name ["mnist", "fashion_mnist", "emnist/balanced"]```: \
    Loads a dataset via TensorFlow datasets (tfds) API. \
    ```--dataset_name ["svhn-7-ex.npz", "cifar10-50-ex.npz"]```: \
    Loads a dataset via numPy. \
    ```--dataset_load ["tfds", "from_npz"]```: \
    Use `"tfds"` when loading MNIST, Fashion-MNIST and E-MNIST. Use `"from_npz"` when using your own datasets (in the numpy archive formats `.npz`or `.npy`), such as latent encoded SVHN and CIFAR-10.

2) **Task settings:** \
    ```--DAll 0 1 2 3 4 5 6 7 8 9 ```: \
    This list should contain all classes that are considered for the training and testing procedures. \
    ```--num_tasks 4 ```: \
    Set this to the number of total task to perform. \
    ```--TX 4 2```: \
    Specify the task sequence in this manner, each identifier is followed by the classes contained in the task. E.g., `--T1 1 2 3 --T2 4 5 --T3 6 7 8 9`. 

3) **Some examples of how to set up a specific model:**
   1) AR, DGR & ER loss coefficients: 
      - We can define the loss weighting strategy with ```--loss_coef ["off", "class_balanced", "task_balanced"]```. This will dynamically calculate the loss coefficients applied to each sample. 
   2) DGR:
      - These settings are used for a balanced replay strategy: ```--replay_proportions -1. -1 --samples_to_generate 1.``` 
      - These settings are used for a constant replay strategy: ```--replay_proportions 50. 50. --samples_to_generate -1.``` 
      - To drop the generator after each task: ```--drop_generator ["no", "yes"]``` 
      - To drop the solver after each task: ```--drop_solver ["no", "yes"]``` 
   3) ER:
      - This adjusts ER's buffer capacity and how many samples per class are saved after each consecutive sub-task: \
      ```--storage_budget: 1500 --per_class_budget: 50```
   4) RTS: \
   To activate "Replay-To-Suppress" you need to specify the following settings:
   ```
    --forgetting_mode               [separate, mixed] \
    --forgetting_tasks              2 4               \
    --extra_eval                    3 5 6 7           \
   ```
    The first option determines whether forgetting takes place in a separate, dedicated removal phase or is interleaved with the acquisition of new task data. In addition, we define which tasks are "to be suppressed" and finally select the classes we want to evaluate additionally to measure knowledge retention.

    To use "Selective Amnesia" based on the C-VAE implementation, you need to setup the arguments accordingly:
    ```
    --amnesiac                      yes   \
    --sa_forg_iters                 10000 \
    --sa_fim_samples                10000 \
    --sa_lambda                     100.  \
    --sa_gamma                      1.0   \
    ```

### Visualize Gaussian densities (AR):
To visualize Gaussian components for AR you can run the following script to generate an image showcasing the trained densities. Please adjust the `--sequence_path` and `--out` arguments as required. 
```
python3 "${PROJ_PATH}/src/cl_replay/api/utils/plot/vis_protos.py"       \
    --sequence_path "${SAVE_DIR}/protos/${EXP_ID}_protos_TX"            \
    --prefix "${EXP_ID}_LX_GMM_"                                        \
    --out "${SAVE_DIR}/protos/visualized/protos_TX"                     \
    --epoch -1                                                          \
    --channels 1                                                        \
    --proto_size 28 28                                                  \
    --pad   " 0.1 "                                                     \
    --h_pad " 0."                                                       \
    --w_pad " -10."                                                     ;
```

## Description of the experimental API:
```
├── api
│   ├── callback
│   │   ├── Log_Metrics.py
│   │   └── Manager.py
│   ├── checkpointing
│   │   └── Manager.py
│   ├── data
│   │   ├── encoding
|   │   │   ├── ContrastiveTrainer.py
|   │   │   ├── DatasetEncoder.py
|   │   │   └── simclr_trainer.py
│   │   ├── Dataset.py
│   │   └── Sampler.py
│   ├── experiment
│   │   ├── adaptor
│   │   │   └── Supervised_Replay_Adaptor.py
│   │   ├── Experiment.py
│   │   └── Experiment_Replay.py
│   ├── layer
│   │   ├── keras
│   │   │   ├── BatchNorm_Layerr.py
|   |   |   ├── ....
│   │   │   └── Reshape_Layer.py
│   │   └── Layer.py
│   ├── model
|   |   ├── DNN.py
│   │   └── Func_Model.py
│   ├── parsing
│   │   ├── Command_Line_Parser.py
│   │   └── Kwarg_Parser.py
│   └── utils
|   |   └── ...
├── architecture
│   ├── ar
│   ├── dgr
│   ├── ewc
│   ├── rehearsal
│   └── sa
```

- This API follows an _"experiment - model - layer"_ design to allow experimentation with different ML architectures. The aim is to provide an easy to use and extensible codebase based on Python subclassing and the [Keras functional paradigm](https://keras.io/guides/functional_api/).
- We use this framework heavily, to investigate continual learning (e.g. class-incremental learning) withh architecttures based on the concept of _generative replay (GR)_, as shown in our recent publications ["Adiabatic Replay for Continual Learning"](https://ieeexplore.ieee.org/abstract/document/10651381), and ["An analysis of best-practice strategies for replay and rehearsal in continual learning"](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Krawczyk_An_Analysis_of_Best-practice_Strategies_for_Replay_and_Rehearsal_in_CVPRW_2024_paper.pdf). We also provide algorithms for regularisation-based approaches such as Elastic Weight Consolidation (EWC) or constraint-based optimisation, i.e. Gradient Episodic Memory (GEM).
- The focus is clearly on supervised image classification, but it is by no means limited to this assumption. This API and its architectural packages can also be seamlessly integrated into other frameworks to investigate other scenarios, such as reinforcement learning (RL), as demonstrated in this work: ["Continual Reinforcement Learning Without Replay Buffers"](https://ieeexplore.ieee.org/abstract/document/10705256).
- Most of the code is executed pythonically, i.e. in TF _"eager execution"_ mode. However, a few specific, computationally expensive functions may or can be annotated with the `@tf.function` decorator to take advantage of TFs graph mode.
- The _"Experimental API"_ package `./api` contains all the essentials for empirical evaluation of arbitrary CL algorithms defined by a custom model and its internal layers.

**Experiment**
- Each _"Experiment class"_ defines how the data is prepared and processed by some ML model in a particular learning scenario (e.g. class-incremental learning), and thus acts as an interface to bind all necessary components of the _"main API"_ to the specific CL algorithm applied to the learning problem.
- The code defining each experiment is written in plain Python using some well-known libraries such as NumPy. Some specific calls, mainly those that run the model, use Keras-specific functions, e.g., `fit()`, `train_on_batch()`, `eval()`, andd `test_step()`.
- We provide to ways to train and evaluate a model: 
  1) By calling `model.fit()` we rely on Keras' automatic training procedure.
  2) By defining a custom training loop, we can perform repeated calls to `model.train_on_batch()` and `model.test_step()`.
* Existing experiments can be easily extended using Python subclassing. This allows the creation of highly complex & customisable training strategies by building on existing implementations, see for example `.architecture.ar.experiment.Experiment_AR`.

**Data Handling**
- SCCL uses an input pipeline that supports both TensorFlow and NumPy to pre-process, manipulate and consume data via the `.api/data/Dataset` class. This class takes care of any necessary data conversion, normalization, and other preparation such as splitting the data into distinct sets.
- Data is provided as instances of `tensorflow.Tensor` or `numpy.ndarray`, specified by the `--data_type` parameter.
- The API loads from either predefined TF datasets (tfds API) or custom numpy archives by setting `--dataset_load` to either `tfds` or `from_npz`.
- For TF, the convenient `tensorflow_datasets` API allows automated import and preparation of commonly used benchmarks (such as MNIST) that can be natively consumed by any TF or Keras implementation. Specific datasets can be selected using the `--dataset_name` parameter, e.g. `MNIST` or `Fashion_MNIST`.
- Usually it should be sufficient to use the prepared data directly in the experimental pipeline, but sometimes it is necessary to have fine-grained control over the class/task mixing ratio, as well as to integrate artificial data from a generative model. For this specific reason, we rely on an instance of the `_sampler_` class to mix and merge data from different sources.
- Data is often divided into well-defined tasks that are usually processed sequentially (class-incremental or task-incremental learning), but we do not prohibit the use of streaming data in any way, the only requirement is that the data comes in the required form and matches what the model expects to process. However, we do not provide an additional interface for streaming data at this time.
- Optional data encoding can be done using the `.api.data.encoding.DataSetEncoder.py` class. This class implements methods for feature extraction, e.g., self-supervised contrastive learning ([SupCon](https://arxiv.org/pdf/2004.11362)) such as [SimCLR](https://arxiv.org/abs/2002.05709).

**Models & Layers**
- Each _"model"_, e.g., `architecture.ar.model.DCGMM.py` defines a specific unsupervised or supervised ML model, which is a sub-class derived from our custom functional model defined by `api.model.Func_Model.py`. 
- A functional model inherits from `keras.models.Model` and extends it in a way to fit 
our custom experimental pipeline.
- In fact, you can simply override the abstract methods to define a custom training loop and support customization at a fine-grained level, such as the training step defined by `DCGMM.train_step()`.
- _"Layers"_ can either be Keras built-ins, e.g., `keras.layers.Input`, extended Keras built-ins, e.g., `api.layer.keras.Input_Layer.py`, or are defined in the same manner as a sub-classed model, e.g., `architecture.ar.layer.GMM_Layer.py.`.
- A custom layer inherits from the base `Layer` class, which in turn inherits from `keras.layers.Layer`. Therefore, each layer implements a concrete logic that usually expects an input `tf.Tensor`, performs some computation based on the input tensor, and as a result returns a transformed output tensor. 
- Often, defined layers expect a shape argument to build up the necessary internal structures, which depends on the dimensionality of the input tensor. 
- This modular _"layer block approach"_ allows the combination of various layer objects to define an arbitrary functional model. We support simple linear topologies, as well as multi-branched architectures, where layers do not necessarily have to be chained in a linear manner.
- For more details about the sub-classing API, please refer to this [link](https://www.tensorflow.org/guide/keras/custom_layers_and_models).

**Command-Line/Argument Parsing**
- Experiments, models & layers are configurable by passing specific parameters via the command line parser or directly within the class's `__init__` method.
- The argument parser provides bash-style support for defining, configuring, and running specific experiments on implemented CL algorithms and their add-on modules.
- Parameters specific to each class can be looked up in the respective source file, see the `_init_parser()` function.
  - For example, loading a specific model architecture can be achieved by adjusting the package path `--model_path <PATH_TO_MODEL>`, to e.g. `cl_replay.architecture.ar.model.DCGMM`.
  - A layer hierarchy can then be defined by providing a plausible chain of layers with the argument `--L<POS> <cl_replay.architecture.ar.layer.LAYER_NAME>`, specifying the layer type, e.g. `Folding_Layer`, `GMM_Layer`, `Readout_Layer`. These layers are either predefined package modules or can be added externally.
  - Most parameters can be manipulated to define the behaviour of a layer: `--L<POS>_<PARAM NAME> <VAL>`.
  - Layers are defined from lowest (input) to highest (output). For example, the argument `--L2 ${PATH_TO_AR_MODULE}.layer.GMM_Layer`, would set the models layer at position 2 to be a GMM layer. Please make sure that layers are connected to each other and correctly specify the arguments `--model_inputs` and `--model_outputs` to match the topology. To configure a layer-specific parameter, we can add a line with `--L2_K 100`, to set the number of Gaussian components.
- Note: It is necessary to set the `$PYTHONPATH` environment variable corresponding to the modules you want to import, so that the Python interpreter is able to find them!
- Another Note: Running experiments via the predefined bash files (on any linux OS) is not mandatory. A functional keras model may also be defined programmatically. The file `./test/Test.py` contains some minimalistic examples on how to create and use some of the pre-defined custom or keras sub-classes directly in your source code, for more details please refer to the [functional API](https://keras.io/guides/functional_api/).
  
**Checkpointing**
- To create checkpoints after each sequential task, use the `--save_all` parameter.
- To load a model from a checkpoint, use `--load_task <TASK>`.
- Define the checkpoint directory with `--ckpt_dir` (default=`~/chkpts/`) and the file prefix `--exp_id <ID>`. 
- The naming conventions for created checkpoint files are `<ID><MODEL_NAME><TASK>.chkpt`.

**Callbacks**
- Callbacks can be attached to a built model instance by passing separate lists of training, evaluation or global callback modules.
- Each callback implements unique functionality that is called on specific events in the timeline, e.g. `on_train_begin()`, `on_epoch_end()`.
- A custom callback can be easily integrated or customized and has access to the compiled model instance. 
- Keras will execute it in an automated way when you use `model.fit()` or `model.evaluate()`.
- However, if a custom training or evaluation loop is used, the callbacks must be called manually, e.g., by directly using the callback functions, such as `cb.on_epoch_end()`. The custom training loops provided in the experimental classes already support this functionality, have a look at them if you need to adapt your own code.

**Experimental Metrics**
- A model and each of it's layers can produce different training or evaluation metrics, depending on its use case.
- A `Readout_Layer` can be evaluated by the results of its loss function (e.g., softmax cross-entropy or MSE), as well as its classification accuracy based on the resulting logits.
- This must be taken into account during the model build, training, and evaluation phases. The model aggregates all internal layer metrics over epochs, then updates the internal data structure after each training or evaluation phase, and finally flushes them to the file system via the `api.callback.Log_Metrics.py` class upon completion.

**Debug/Info Logging**
- The Python logging module is active and prints messages for live evaluation.
- Log levels can be set by specifying `--log_level <LEVEL>` with the values: `['DEBUG', 'INFO']`.
- If additional information is needed for a particular class, the `api.utils.log` package can simply be imported and the code can be annotated using `log.info('info message')` or `log.debug('debug msg')`.
- Note that code executed in TF's graph mode is printed only once during the graph tracing phase. You might consider using `tf.print('msg')` within functions annotated with `@tf.function` or when running in graph mode.
