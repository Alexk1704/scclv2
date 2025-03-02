BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="er-emnist-replay"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets"
AR_MOD="cl_replay.architecture.ar"
LABEL_DIM=47
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.rehearsal.experiment.Experiment_Rehearsal \
--exp_id                        "${EXP_ID}"                     \
--log_level                     DEBUG                           \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_load                  tfds                            \
--dataset_name                  emnist/balanced                 \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     6                               \
--DAll                          0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
--T1                            5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
--T2                            0                               \
--T3                            1                               \
--T4                            2                               \
--T5                            3                               \
--T6                            4                               \
--num_classes                   "${LABEL_DIM}"                  \
--epochs                        100                             \
--batch_size                    128                             \
--test_batch_size               128                             \
--load_task                     0                               \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    dnn                             \
--num_layers                    3                               \
--num_units                     512                             \
--add_dropout                   no                              \
--dropout_rate                  0.3                             \
--freeze_n_layers               0                               \
--opt                           sgd                             \
--sgd_epsilon                   1e-3                            \
--sgd_momentum                  0.                              \
--sgd_wdecay                    5e-4                            \
--adam_epsilon                  1e-4                            \
--adam_beta1                    .90                             \
--adam_beta2                    .999                            \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR}"                   \
--vis_path                      "${SAVE_DIR}/vis"               \
--ckpt_dir                      "${SAVE_DIR}"                   \
--log_training                  no                              \
--replay_proportions            50. 50.                         \
--samples_to_generate           -1.                             \
--loss_coef                     off                             \
--budget_method                 class                           \
--storage_budget                2000                            \
--per_class_budget              50                              \
--per_task_budget               .005                            \
--rehearsal_type                batch                           \
--balance_type                  tasks             