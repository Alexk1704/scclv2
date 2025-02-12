BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/git/scclv2/src
EXP_ID="ar-dcgmm-3l"
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
--dataset_name                  mnist                           \
--dataset_load                  tfds                            \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--vis_batch                     no                              \
--vis_gen                       no                              \
--data_type                     32                              \
--num_tasks                     2                               \
--DAll                          0 4 6 9                         \
--T1                            0 4 6                           \
--T2                            9                               \
--epochs                        128                             \
--batch_size                    64                              \
--test_batch_size               64                              \
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
--patience                      128                                 \
--sampling_batch_size           64                                  \
--samples_to_generate           1.                                  \
--sampling_layer                -1                                  \
--sample_variants               yes                                 \
--sample_topdown                no                                  \
--replay_proportions            50. 50.                             \
--loss_coef                     off                                 \
--loss_masking                  no                                  \
--alpha_wrong                   1.                                  \
--alpha_right                   .01                                 \
--ro_layer_index                3                                   \
--ro_patience                   -1                                  \
--model_inputs                  0                                   \
--model_outputs                 11                                  \
--sampling_branch               11 10 9 6 5 4 3 2 1 0               \
--L0                        Input_Layer \
--L0_layer_name             L0_INPUT    \
--L0_shape                  28 28 1     \
--L1                        ${AR_MOD}.layer.Folding_Layer \
--L1_layer_name             L1_FOLD1                      \
--L1_patch_width            3                             \
--L1_patch_height           3                             \
--L1_stride_x               1                             \
--L1_stride_y               1                             \
--L1_sharpening_iterations  0                             \
--L1_sharpening_rate        0.0                           \
--L1_input_layer            0                             \
--L2                        ${AR_MOD}.layer.GMM_Layer \
--L2_layer_name             L2_GMM1                   \
--L2_K                      25                        \
--L2_conv_mode              yes                       \
--L2_sampling_divisor       10                        \
--L2_sampling_I             -1                        \
--L2_sampling_S             3                         \
--L2_sampling_P             1.                        \
--L2_somSigma_sampling      no                        \
--L2_eps_0                  0.011                     \
--L2_eps_inf                0.01                      \
--L2_lambda_sigma           0.                        \
--L2_lambda_pi              0.                        \
--L2_reset_factor           0.1                       \
--L2_gamma                  0.90                      \
--L2_alpha                  0.01                      \
--L2_loss_masking           no                        \
--L2_input_layer            1                         \
--L3                        ${AR_MOD}.layer.Folding_Layer \
--L3_layer_name             L3_FOLD2                      \
--L3_patch_width            4                             \
--L3_patch_height           4                             \
--L3_stride_x               2                             \
--L3_stride_y               2                             \
--L3_sharpening_iterations  0                             \
--L3_sharpening_rate        0.0                           \
--L3_input_layer            2                             \
--L4                        ${AR_MOD}.layer.GMM_Layer \
--L4_layer_name             L4_GMM2                   \
--L4_K                      25                        \
--L4_conv_mode              yes                       \
--L4_sampling_divisor       10                        \
--L4_sampling_I             -1                        \
--L4_sampling_S             3                         \
--L4_sampling_P             1.                        \
--L4_somSigma_sampling      no                        \
--L4_eps_0                  0.011                     \
--L4_eps_inf                0.01                      \
--L4_lambda_sigma           0.                        \
--L4_lambda_pi              0.                        \
--L4_reset_factor           0.1                       \
--L4_gamma                  0.90                      \
--L4_alpha                  0.01                      \
--L4_loss_masking           no                        \
--L4_wait_target            L2                        \
--L4_wait_threshold         2.0                       \
--L4_input_layer            3                         \
--L5                        ${AR_MOD}.layer.Folding_Layer \
--L5_layer_name             L5_FOLD2                      \
--L5_patch_width            12                            \
--L5_patch_height           12                            \
--L5_stride_x               1                             \
--L5_stride_y               1                             \
--L5_sharpening_iterations  0                             \
--L5_sharpening_rate        0.0                           \
--L5_input_layer            4                             \
--L6                        ${AR_MOD}.layer.GMM_Layer \
--L6_layer_name             L6_GMM3                   \
--L6_K                      49                        \
--L6_conv_mode              yes                       \
--L6_sampling_divisor       10                        \
--L6_sampling_I             -1                        \
--L6_sampling_S             3                         \
--L6_sampling_P             1.                        \
--L6_somSigma_sampling      no                        \
--L6_eps_0                  0.011                     \
--L6_eps_inf                0.01                      \
--L6_lambda_sigma           0.                        \
--L6_lambda_pi              0.                        \
--L6_reset_factor           0.1                       \
--L6_gamma                  0.90                      \
--L6_alpha                  0.01                      \
--L6_loss_masking           no                        \
--L6_wait_target            L2 L4                     \
--L6_wait_threshold         2.0 2.0                   \
--L6_input_layer            5                         \
--L7                        cl_replay.api.layer.keras.Reshape_Layer \
--L7_layer_name             L7_RESHAPE1                             \
--L7_target_shape           1 1 16900                               \
--L7_input_layer            2                                       \
--L8                        cl_replay.api.layer.keras.Reshape_Layer \
--L8_layer_name             L8_RESHAPE2                             \
--L8_target_shape           1 1 3600                                \
--L8_input_layer            4                                       \
--L9                        cl_replay.api.layer.keras.Reshape_Layer \
--L9_layer_name             L9_RESHAPE3                             \
--L9_target_shape           1 1 49                                  \
--L9_input_layer            6                                       \
--L10                       cl_replay.api.layer.keras.Concatenate_Layer \
--L10_layer_name            L10_CONCAT                                  \
--L10_input_layer           7 8 9                                       \
--L11                       ${AR_MOD}.layer.Readout_Layer \
--L11_layer_name            L11_READOUT                   \
--L11_num_classes           10                            \
--L11_loss_function         mean_squared_error            \
--L11_lambda_b              0.                            \
--L11_regEps                0.05                          \
--L11_loss_masking          no                            \
--L11_reset                 no                            \
--L11_wait_target           L2 L4 L6                      \
--L11_wait_threshold        10.0 10.0 10.0                \
--L11_input_layer           10