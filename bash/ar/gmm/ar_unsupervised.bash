BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="ar-unsupervised"
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
--ml_paradigm                   unsupervised                    \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_name                  lf-straight-5000.npz            \
--dataset_load                  from_npz                        \
--test_split                    0.0                             \
--vis_batch                     no                              \
--vis_gen                       no                              \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     1                               \
--T1                            100                             \
--epochs                        1024                            \
--batch_size                    64                              \
--test_batch_size               64                              \
--load_task                     0                               \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     no                              \
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
--patience                      250                                 \
--sampling_batch_size           50                                  \
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
--model_outputs                 2                                   \
--L0                        Input_Layer  \
--L0_layer_name             L0_INPUT     \
--L0_shape                  12 100 1     \
--L1                        ${SCCL_MOD}.layer.keras.Reshape_Layer   \
--L1_layer_name             L1_RESHAPE                              \
--L1_target_shape           1 1 1200                                \
--L1_prev_shape             12 100 1                                \
--L1_input_layer            0                                       \
--L2                        ${AR_MOD}.layer.GMM_Layer   \
--L2_layer_name             L2_GMM                      \
--L2_K                      256                         \
--L2_conv_mode              yes                         \
--L2_sampling_divisor       10                          \
--L2_sampling_I             -1                          \
--L2_sampling_S             3                           \
--L2_sampling_P             1.                          \
--L2_eps_0                  0.011                       \
--L2_eps_inf                0.01                        \
--L2_lambda_sigma           0.                          \
--L2_lambda_pi              0.                          \
--L2_reset_factor           0.1                         \
--L2_gamma                  0.90                        \
--L2_loss_masking           no                          \
--L2_log_bmu_activity       no                          \
--L2_input_layer            1                           