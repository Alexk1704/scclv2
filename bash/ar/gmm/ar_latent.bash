BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="ar-latent"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.ar.experiment.Experiment_AR \
--exp_id                        "${EXP_ID}"                     \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_name                  cifar10-50-ex.npz               \
--dataset_load                  from_npz                        \
--vis_batch                     no                              \
--vis_gen                       no                              \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     4                               \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4 5 6 7 8 9             \
--T2                            7                               \
--T3                            8                               \
--T4                            9                               \
--epochs                        999                             \
--batch_size                    100                             \
--test_batch_size               100                             \
--load_task                     0                               \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    ${AR_MOD}.model.DCGMM               \
--callback_paths                ${AR_MOD}.callback                  \
--train_callbacks               Set_Model_Params Early_Stop         \
--global_callbacks              Log_Metrics Log_Protos              \
--log_path                      "${SAVE_DIR}"                       \
--vis_path                      "${SAVE_DIR}/protos"                \
--ckpt_dir                      "${SAVE_DIR}"                       \
--log_training                  no                                  \
--save_protos                   on_train_end                        \
--log_each_n_protos             32                                  \
--ro_patience                   no                                  \
--patience                      64                                  \
--sampling_batch_size           100                                 \
--samples_to_generate           1.                                  \
--sampling_layer                -1                                  \
--sample_variants               yes                                 \
--sample_topdown                no                                  \
--replay_task                   supervised                          \
--replay_proportions            50. 50.                             \
--loss_masking                  no                                  \
--alpha_wrong                   1.                                  \
--alpha_right                   .01                                 \
--ro_layer_index                2                                   \
--ro_patience                   -1                                  \
--model_inputs                  0                                   \
--model_outputs                 2                                   \
--L0                        Input_Layer \
--L0_layer_name             L0_INPUT    \
--L0_shape                  1 1 2048    \
--L1                        ${AR_MOD}.layer.GMM_Layer   \
--L1_layer_name             L1_GMM                      \
--L1_K                      225                         \
--L1_conv_mode              yes                         \
--L1_sampling_divisor       10                          \
--L1_sampling_I             -1                          \
--L1_sampling_S             3                           \
--L1_sampling_P             1.                          \
--L1_somSigma_sampling      no                          \
--L1_eps_0                  0.011                       \
--L1_eps_inf                0.01                        \
--L1_lambda_sigma           0.                          \
--L1_lambda_pi              0.                          \
--L1_reset_factor           0.1                         \
--L1_gamma                  0.96                        \
--L1_alpha                  0.011                       \
--L1_loss_masking           no                          \
--L1_log_bmu_activity       no                          \
--L1_input_layer            0                           \
--L2                        ${AR_MOD}.layer.Readout_Layer   \
--L2_layer_name             L2_READOUT                      \
--L2_num_classes            10                              \
--L2_loss_function          mean_squared_error              \
--L2_lambda_b               0.                              \
--L2_regEps                 0.05                            \
--L2_loss_masking           no                              \
--L2_reset                  no                              \
--L2_wait_target            L1                              \
--L2_wait_threshold         100.                            \
--L2_input_layer            1 
