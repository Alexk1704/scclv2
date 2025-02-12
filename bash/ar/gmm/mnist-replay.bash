BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/git/scclv2/src
EXP_ID="mnist-replay-046-9"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
PROJ_PATH="${BASE}/git/scclv2"
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
--vis_batch                     no                              \
--vis_gen                       no                              \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     1                               \
--DAll                          0 4 6 9                         \
--T1                            0 4 6                           \
--T2                            9                               \
--epochs                        100                             \
--batch_size                    128                             \
--test_batch_size               128                             \
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
--sampling_batch_size           128                                 \
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
--model_outputs                 3                                   \
--L0                        Input_Layer \
--L0_layer_name             L0_INPUT    \
--L0_shape                  28 28 1     \
--L1                        ${AR_MOD}.layer.Folding_Layer   \
--L1_layer_name             L1_FOLD                         \
--L1_patch_width            28                              \
--L1_patch_height           28                              \
--L1_stride_x               1                               \
--L1_stride_y               1                               \
--L1_input_layer            0                               \
--L2                        ${AR_MOD}.layer.GMM_Layer   \
--L2_layer_name             L2_GMM                      \
--L2_K                      25                          \
--L2_conv_mode              yes                         \
--L2_sampling_divisor       10                          \
--L2_sampling_I             -1                          \
--L2_sampling_S             3                           \
--L2_sampling_P             1.                          \
--L2_somSigma_sampling      yes                         \
--L2_eps_0                  0.011                       \
--L2_eps_inf                0.01                        \
--L2_lambda_sigma           0.                          \
--L2_lambda_pi              0.                          \
--L2_reset_factor           0.1                         \
--L2_gamma                  0.90                        \
--L2_alpha                  0.01                        \
--L2_loss_masking           no                          \
--L2_log_bmu_activity       no                          \
--L2_input_layer            1                           \
--L3                        ${AR_MOD}.layer.Readout_Layer   \
--L3_layer_name             L3_READOUT                      \
--L3_num_classes            10                              \
--L3_loss_function          mean_squared_error              \
--L3_lambda_b               0.                              \
--L3_regEps                 0.05                            \
--L3_loss_masking           no                              \
--L3_reset                  no                              \
--L3_wait_target            L2                              \
--L3_wait_threshold         100.                            \
--L3_input_layer            2                               ;
${INVOCATION} python3 "${PROJ_PATH}/src/cl_replay/api/utils/plot/vis_protos.py" \
    --sequence_path "${SAVE_DIR}/protos/${EXP_ID}_protos_T1"            \
    --prefix "${EXP_ID}_L2_GMM_"                                        \
    --out "${SAVE_DIR}/protos/visualized/protos_T1"                     \
    --epoch -1                                                          \
    --channels 1                                                        \
    --proto_size 28 28                                                  \
    --pad   " 0.1 "                                                     \
    --h_pad " 0."                                                       \
    --w_pad " -10."                                                     ;
${INVOCATION} python3 -m cl_replay.architecture.ar.experiment.Experiment_AR \
--project_name                  AR                              \
--architecture                  AR-FLAT                         \
--exp_group                     AR-FLAT                         \
--exp_tags                      AR-FLAT VARIANTS                \
--exp_id                        ${EXP_ID}                       \
--wandb_active                  no                              \
--dataset_dir                   ${DATA_DIR}                     \
--dataset_name                  mnist                           \
--dataset_load                  tfds                            \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--vis_batch                     yes                             \
--vis_gen                       yes                             \
--data_type                     32                              \
--num_tasks                     2                               \
--DAll                          0 4 6 9                         \
--T1                            0 4 6                           \
--T2                            9                               \
--epochs                        5                               \
--batch_size                    128                             \
--test_batch_size               128                             \
--log_level                     DEBUG                           \
--load_task                     1                               \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    ${AR_MOD}.model.DCGMM               \
--callback_paths                ${AR_MOD}.callback                  \
--train_callbacks               Set_Model_Params Early_Stop         \
--global_callbacks              Log_Metrics Log_Protos              \
--log_path                      ${SAVE_DIR}                         \
--vis_path                      ${SAVE_DIR}/protos                  \
--ckpt_dir                      ${SAVE_DIR}                         \
--log_training                  no                                  \
--save_protos                   on_train_end                        \
--log_each_n_protos             32                                  \
--ro_patience                   no                                  \
--patience                      128                                 \
--sampling_batch_size           128                                 \
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
--model_outputs                 3                                   \
--L0                        Input_Layer \
--L0_layer_name             L0_INPUT    \
--L0_shape                  28 28 1     \
--L1                        ${AR_MOD}.layer.Folding_Layer   \
--L1_layer_name             L1_FOLD                         \
--L1_patch_width            28                              \
--L1_patch_height           28                              \
--L1_stride_x               1                               \
--L1_stride_y               1                               \
--L1_input_layer            0                               \
--L2                        ${AR_MOD}.layer.GMM_Layer   \
--L2_layer_name             L2_GMM                      \
--L2_K                      25                          \
--L2_conv_mode              yes                         \
--L2_sampling_divisor       10                          \
--L2_sampling_I             -1                          \
--L2_sampling_S             3                           \
--L2_sampling_P             1.                          \
--L2_somSigma_sampling      yes                         \
--L2_eps_0                  0.011                       \
--L2_eps_inf                0.01                        \
--L2_lambda_sigma           0.                          \
--L2_lambda_pi              0.                          \
--L2_reset_factor           0.1                         \
--L2_gamma                  0.90                        \
--L2_alpha                  0.01                        \
--L2_loss_masking           no                          \
--L2_log_bmu_activity       no                          \
--L2_input_layer            1                           \
--L3                        ${AR_MOD}.layer.Readout_Layer   \
--L3_layer_name             L3_READOUT                      \
--L3_num_classes            10                              \
--L3_loss_function          mean_squared_error              \
--L3_lambda_b               0.                              \
--L3_regEps                 0.05                            \
--L3_loss_masking           no                              \
--L3_reset                  no                              \
--L3_wait_target            L2                              \
--L3_wait_threshold         100.                            \
--L3_input_layer            2                               ;
${INVOCATION} python3 "${PROJ_PATH}/src/cl_replay/api/utils/plot/vis_protos.py" \
    --sequence_path "${SAVE_DIR}/protos/${EXP_ID}_protos_T2"            \
    --prefix "${EXP_ID}_L2_GMM_"                                        \
    --out "${SAVE_DIR}/protos/visualized/protos_T2-R0.1"                \
    --epoch -1                                                          \
    --channels 1                                                        \
    --proto_size 28 28                                                  \
    --pad   " 0.1 "                                                     \
    --h_pad " 0."                                                       \
    --w_pad " -10."                                                     ;
# ${INVOCATION} python3 -m cl_replay.architecture.ar.experiment.Experiment_AR \
# --project_name                  AR                              \
# --architecture                  AR-FLAT                         \
# --exp_group                     AR-FLAT                         \
# --exp_tags                      AR-FLAT VARIANTS                \
# --exp_id                        ${EXP_ID}                       \
# --wandb_active                  no                              \
# --dataset_dir                   ${DATA_DIR}                     \
# --dataset_name                  mnist                           \
# --dataset_load                  tfds                            \
# --renormalize01                 yes                             \
# --np_shuffle                    yes                             \
# --vis_batch                     yes                             \
# --vis_gen                       yes                             \
# --data_type                     32                              \
# --num_tasks                     2                               \
# --DAll                          0 4 6 9                         \
# --T1                            0 4 6                           \
# --T2                            9                               \
# --epochs                        5                               \
# --batch_size                    128                             \
# --test_batch_size               128                             \
# --log_level                     DEBUG                           \
# --load_task                     1                               \
# --save_All                      yes                             \
# --train_method                  fit                             \
# --test_method                   eval                            \
# --full_eval                     yes                             \
# --single_class_test             no                              \
# --model_type                    ${AR_MOD}.model.DCGMM               \
# --callback_paths                ${AR_MOD}.callback                  \
# --train_callbacks               Set_Model_Params Early_Stop         \
# --global_callbacks              Log_Metrics Log_Protos              \
# --log_path                      ${SAVE_DIR}                         \
# --vis_path                      ${SAVE_DIR}/protos                  \
# --ckpt_dir                      ${SAVE_DIR}                         \
# --log_training                  no                                  \
# --save_protos                   on_train_end                        \
# --log_each_n_protos             32                                  \
# --ro_patience                   no                                  \
# --patience                      128                                 \
# --sampling_batch_size           128                                 \
# --samples_to_generate           1.                                  \
# --sampling_layer                -1                                  \
# --sample_variants               yes                                 \
# --sample_topdown                no                                  \
# --replay_proportions            50. 50.                             \
# --loss_coef                     off                                 \
# --loss_masking                  no                                  \
# --alpha_wrong                   1.                                  \
# --alpha_right                   .01                                 \
# --ro_layer_index                3                                   \
# --ro_patience                   -1                                  \
# --model_inputs                  0                                   \
# --model_outputs                 3                                   \
# --L0                        Input_Layer \
# --L0_layer_name             L0_INPUT    \
# --L0_shape                  28 28 1     \
# --L1                        ${AR_MOD}.layer.Folding_Layer   \
# --L1_layer_name             L1_FOLD                         \
# --L1_patch_width            28                              \
# --L1_patch_height           28                              \
# --L1_stride_x               1                               \
# --L1_stride_y               1                               \
# --L1_input_layer            0                               \
# --L2                        ${AR_MOD}.layer.GMM_Layer   \
# --L2_layer_name             L2_GMM                      \
# --L2_K                      25                          \
# --L2_conv_mode              yes                         \
# --L2_sampling_divisor       10                          \
# --L2_sampling_I             -1                          \
# --L2_sampling_S             3                           \
# --L2_sampling_P             1.                          \
# --L2_somSigma_sampling      yes                         \
# --L2_eps_0                  0.011                       \
# --L2_eps_inf                0.01                        \
# --L2_lambda_sigma           0.                          \
# --L2_lambda_pi              0.                          \
# --L2_reset_factor           0.5                         \
# --L2_gamma                  0.90                        \
# --L2_alpha                  0.01                        \
# --L2_loss_masking           no                          \
# --L2_log_bmu_activity       no                          \
# --L2_input_layer            1                           \
# --L3                        ${AR_MOD}.layer.Readout_Layer   \
# --L3_layer_name             L3_READOUT                      \
# --L3_num_classes            10                              \
# --L3_loss_function          mean_squared_error              \
# --L3_lambda_b               0.                              \
# --L3_regEps                 0.05                            \
# --L3_loss_masking           no                              \
# --L3_reset                  no                              \
# --L3_wait_target            L2                              \
# --L3_wait_threshold         100.                            \
# --L3_input_layer            2                               ;
# ${INVOCATION} python3 "${PROJ_PATH}/src/cl_replay/api/utils/plot/vis_protos.py" \
#     --sequence_path "${SAVE_DIR}/protos/${EXP_ID}_protos_T2"            \
#     --prefix "${EXP_ID}_L2_GMM_"                                        \
#     --out "${SAVE_DIR}/protos/visualized/protos_T2-R0.5"                \
#     --epoch -1                                                          \
#     --channels 1                                                        \
#     --proto_size 28 28                                                  \
#     --pad   " 0.1 "                                                     \
#     --h_pad " 0."                                                       \
#     --w_pad " -10."                                                     ;
# python3 -m cl_replay.architecture.ar.experiment.Experiment_AR   \
# --project_name                  AR                              \
# --architecture                  AR-FLAT                         \
# --exp_group                     AR-FLAT                         \
# --exp_tags                      AR-FLAT VARIANTS                \
# --exp_id                        ${EXP_ID}                       \
# --wandb_active                  no                              \
# --dataset_dir                   ${DATA_DIR}                     \
# --dataset_name                  mnist                           \
# --dataset_load                  tfds                            \
# --renormalize01                 yes                             \
# --np_shuffle                    yes                             \
# --vis_batch                     yes                             \
# --vis_gen                       yes                             \
# --data_type                     32                              \
# --num_tasks                     2                               \
# --DAll                          0 4 6 9                         \
# --T1                            0 4 6                           \
# --T2                            9                               \
# --epochs                        5                               \
# --batch_size                    128                             \
# --test_batch_size               128                             \
# --log_level                     DEBUG                           \
# --load_task                     1                               \
# --save_All                      yes                             \
# --train_method                  fit                             \
# --test_method                   eval                            \
# --full_eval                     yes                             \
# --single_class_test             no                              \
# --model_type                    ${AR_MOD}.model.DCGMM               \
# --callback_paths                ${AR_MOD}.callback                  \
# --train_callbacks               Set_Model_Params Early_Stop         \
# --global_callbacks              Log_Metrics Log_Protos              \
# --log_path                      ${SAVE_DIR}                         \
# --vis_path                      ${SAVE_DIR}/protos                  \
# --ckpt_dir                      ${SAVE_DIR}                         \
# --log_training                  no                                  \
# --save_protos                   on_train_end                        \
# --log_each_n_protos             32                                  \
# --ro_patience                   no                                  \
# --patience                      128                                 \
# --sampling_batch_size           128                                 \
# --samples_to_generate           1.                                  \
# --sampling_layer                -1                                  \
# --sample_variants               yes                                 \
# --sample_topdown                no                                  \
# --replay_proportions            50. 50.                             \
# --loss_coef                     off                                 \
# --loss_masking                  no                                  \
# --alpha_wrong                   1.                                  \
# --alpha_right                   .01                                 \
# --ro_layer_index                3                                   \
# --ro_patience                   -1                                  \
# --model_inputs                  0                                   \
# --model_outputs                 3                                   \
# --L0                        Input_Layer \
# --L0_layer_name             L0_INPUT    \
# --L0_shape                  28 28 1     \
# --L1                        ${AR_MOD}.layer.Folding_Layer   \
# --L1_layer_name             L1_FOLD                         \
# --L1_patch_width            28                              \
# --L1_patch_height           28                              \
# --L1_stride_x               1                               \
# --L1_stride_y               1                               \
# --L1_input_layer            0                               \
# --L2                        ${AR_MOD}.layer.GMM_Layer   \
# --L2_layer_name             L2_GMM                      \
# --L2_K                      25                          \
# --L2_conv_mode              yes                         \
# --L2_sampling_divisor       10                          \
# --L2_sampling_I             -1                          \
# --L2_sampling_S             3                           \
# --L2_sampling_P             1.                          \
# --L2_somSigma_sampling      yes                         \
# --L2_eps_0                  0.011                       \
# --L2_eps_inf                0.01                        \
# --L2_lambda_sigma           0.                          \
# --L2_lambda_pi              0.                          \
# --L2_reset_factor           1.0                         \
# --L2_gamma                  0.90                        \
# --L2_alpha                  0.01                        \
# --L2_loss_masking           no                          \
# --L2_log_bmu_activity       no                          \
# --L2_input_layer            1                           \
# --L3                        ${AR_MOD}.layer.Readout_Layer   \
# --L3_layer_name             L3_READOUT                      \
# --L3_num_classes            10                              \
# --L3_loss_function          mean_squared_error              \
# --L3_lambda_b               0.                              \
# --L3_regEps                 0.05                            \
# --L3_loss_masking           no                              \
# --L3_reset                  no                              \
# --L3_wait_target            L2                              \
# --L3_wait_threshold         100.                            \
# --L3_input_layer            2                               ;
# ${INVOCATION} python3 "${PROJ_PATH}/src/cl_replay/api/utils/plot/vis_protos.py" \
#     --sequence_path "${SAVE_DIR}/protos/${EXP_ID}_protos_T2"            \
#     --prefix "${EXP_ID}_L2_GMM_"                                        \
#     --out "${SAVE_DIR}/protos/visualized/protos_T2-R1.0"                \
#     --epoch -1                                                          \
#     --channels 1                                                        \
#     --proto_size 28 28                                                  \
#     --pad   " 0.1 "                                                     \
#     --h_pad " 0."                                                       \
#     --w_pad " -10."                                                     ;
