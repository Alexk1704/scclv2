BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/git/scclv2/src
EXP_ID="ewc-mnist-rts"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.ewc.experiment.Experiment_EWC \
--exp_id                        "${EXP_ID}"                     \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_load                  tfds                            \
--dataset_name                  fashion_mnist                   \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     7                               \
--forgetting_mode               mixed                           \
--forgetting_tasks              2 4 6                           \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4 5                     \
--T2                            0 1 2                           \
--T3                            6 7                             \
--T4                            3 4                             \
--T5                            8                               \
--T6                            5                               \
--T7                            9                               \
--extra_eval                    6 7 8 9                         \
--num_classes                   10                              \
--epochs                        100                             \
--batch_size                    128                             \
--test_batch_size               128                             \
--load_task                     0                               \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    cnn                             \
--num_layers                    3                               \
--num_units                     400                             \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR:?}"                 \
--vis_path                      "${SAVE_DIR:?}/vis"             \
--ckpt_dir                      "${SAVE_DIR:?}"                 \
--log_training                  no                              \
--opt                           adam                            \
--adam_epsilon                  0.0001                          \
--adam_beta1                    0.9                             \
--adam_beta2                    0.999                           \
--sgd_epsilon                   0.00001                         \
--sgd_momentum                  0.0                             \
--sgd_wdecay                                                    \
--loss_coef                     off                             \
--mode                          ewc                             \
--lambda                        100                             \
--imm_transfer_type             weight_transfer                 \
--imm_alpha                     0.5