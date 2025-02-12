BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/git/scclv2/src
EXP_ID="er-latent-rts"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.rehearsal.experiment.Experiment_Rehearsal \
--exp_id                        ${EXP_ID}                       \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_load                  from_npz                        \
--dataset_name                  svhn-7-ex.npz                   \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     5                               \
--forgetting_mode               separate                        \
--forgetting_tasks              2 4                             \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3                         \
--T2                            0 1                             \
--T3                            4 5                             \
--T4                            2 4                             \
--T5                            6 7                             \
--extra_eval                    3 5 6 7                         \
--epochs                        10                              \
--batch_size                    128                             \
--test_batch_size               128                             \
--load_task                     0                               \
--save_All                      yes                             \
--train_method                  batch                           \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    dnn                             \
--num_layers                    3                               \
--num_units                     400                             \
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
--budget_method                 class                     \
--storage_budget                1000                      \
--per_class_budget              50                        \
--per_task_budget               .005                      \
--rehearsal_type                batch                     \
--balance_type                  tasks      
