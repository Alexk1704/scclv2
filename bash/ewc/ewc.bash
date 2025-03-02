BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="ewc-mnist"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.ewc.experiment.Experiment_EWC \
--exp_id                        ${EXP_ID}                       \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_name                  emnist_balanced.npz             \
--dataset_load                  from_npz                        \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     5                               \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1                             \
--T2                            2 3                             \
--T3                            4 5                             \
--T4                            6 7                             \
--T5                            8 9                             \
--num_classes                   47                              \
--epochs                        20                              \
--batch_size                    128                             \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--model_type                    dnn                             \
--num_layers                    3                               \
--num_units                     800                             \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR}"                   \
--vis_path                      "${SAVE_DIR}/vis"               \
--ckpt_dir                      "${SAVE_DIR}"                   \
--log_training                  no                              \
--save_when                     train_end                       \
--opt                           adam                            \
--adam_epsilon                  0.000001                        \
--adam_beta1                    0.9                             \
--adam_beta2                    0.999                           \
--sgd_epsilon                   0.00001                         \
--sgd_momentum                  0.0                             \
--sgd_wdecay                                                    \
--loss_coef                     off                             \
--mode                          ewc                             \
--lambda                        100000                          \
--imm_transfer_type             weight_transfer                 \
--imm_alpha                     0.5