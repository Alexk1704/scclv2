BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="sequential"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
# export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.rehearsal.experiment.Experiment_Sequential \
--exp_id                        ${EXP_ID}                       \
--log_level                     DEBUG                           \
--random_seed                   47                              \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_name                  fruits.npz                      \
--dataset_load                  from_npz                        \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     1                               \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4 5 6 7 8 9             \
--epochs                        200                             \
--batch_size                    128                             \
--test_batch_size               128                             \
--load_task                     0                               \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    cnn                             \
--num_layers                    4                               \
--num_units                     800                             \
--add_dropout                   no                              \
--dropout_rate                  0.3                             \
--freeze_n_layers               2                               \
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
--log_training                  no

