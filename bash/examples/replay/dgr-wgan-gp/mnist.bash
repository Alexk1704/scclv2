BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="dgr-wgangp-mnist-replay"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets"
AR_MOD="cl_replay.architecture.ar"
NOISE_DIM=100
DATA_DIM=784
LABEL_DIM=10
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.dgr.experiment.Experiment_DGR \
--exp_id                        "${EXP_ID}"                     \
--log_level                     DEBUG                           \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_load                  tfds                            \
--dataset_name                  mnist                           \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--vis_batch                     no                              \
--vis_gen                       no                              \
--data_type                     32                              \
--num_tasks                     4                               \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4 5 6                   \
--T2                            7                               \
--T3                            8                               \
--T4                            9                               \
--num_classes                   ${LABEL_DIM}                    \
--batch_size                    128                             \
--sampling_batch_size           128                             \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    DGR-GAN                         \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR}"                   \
--vis_path                      "${SAVE_DIR}"                   \
--ckpt_dir                      "${SAVE_DIR}"                   \
--log_training                  no                              \
--input_size                    28 28 1                         \
--generator_type                GAN                             \
--gan_epochs                    100                             \
--solver_epochs                 50                              \
--solver_epsilon                0.001                           \
--replay_proportions            50. 50.                         \
--samples_to_generate           -1.                             \
--loss_coef                     off                             \
--noise_dim                     ${NOISE_DIM}                    \
--conditional                   no                              \
--wasserstein                   yes                             \
--gp_weight                     10                              \
--wgan_disc_iters               3                               \
--drop_solver                   no                              \
--drop_generator                no                              \
--gan_epsilon                   0.0005                          \
--gan_beta1                     0.5                             \
--gan_beta2                     0.999                           \
--G_model_inputs            0               \
--G_model_outputs           7               \
--G0                        Input_Layer     \
--G0_layer_name             G0_INPUT        \
--G0_shape                  ${NOISE_DIM}    \
--G1                        cl_replay.api.layer.keras.Dense_Layer \
--G1_layer_name             G1_DENSE        \
--G1_units                  2048            \
--G1_activation             none            \
--G1_use_bias               no              \
--G1_input_layer            0               \
--G2                        cl_replay.api.layer.keras.BatchNorm_Layer \
--G2_layer_name             G2_BATCHNORM    \
--G2_input_layer            1               \
--G3                        cl_replay.api.layer.keras.LeakyReLU_Layer \
--G3_layer_name             G3_LEAKYRELU    \
--G3_alpha                  0.2             \
--G3_input_layer            2               \
--G4                        cl_replay.api.layer.keras.Dense_Layer \
--G4_layer_name             G4_DENSE        \
--G4_units                  2048            \
--G4_activation             none            \
--G4_input_layer            3               \
--G5                        cl_replay.api.layer.keras.BatchNorm_Layer \
--G5_layer_name             G5_BATCHNORM    \
--G5_input_layer            4               \
--G6                        cl_replay.api.layer.keras.LeakyReLU_Layer \
--G6_layer_name             G6_LEAKYRELU    \
--G6_alpha                  0.2             \
--G6_input_layer            5               \
--G7                        cl_replay.api.layer.keras.Dense_Layer \
--G7_layer_name             G7_OUT          \
--G7_units                  ${DATA_DIM}     \
--G7_activation             sigmoid         \
--G7_input_layer            6               \
--D_model_inputs            0                   \
--D_model_outputs           7                   \
--D0                        Input_Layer         \
--D0_layer_name             D0_INPUT            \
--D0_shape                  ${DATA_DIM}         \
--D1                        cl_replay.api.layer.keras.Dense_Layer \
--D1_layer_name             D1_DENSE            \
--D1_units                  512                 \
--D1_activation             none                \
--D1_input_layer            0                   \
--D2                        cl_replay.api.layer.keras.LeakyReLU_Layer \
--D2_layer_name             D2_LEAKYRELU        \
--D2_alpha                  0.2                 \
--D2_input_layer            1                   \
--D3                        cl_replay.api.layer.keras.Dropout_Layer \
--D3_rate                   0.3                 \
--D3_layer_name             D3_DROPOUT          \
--D3_input_layer            2                   \
--D4                        cl_replay.api.layer.keras.Dense_Layer \
--D4_layer_name             D4_DENSE            \
--D4_units                  256                 \
--D4_activation             none                \
--D4_input_layer            3                   \
--D5_alpha                  0.2                 \
--D5                        cl_replay.api.layer.keras.LeakyReLU_Layer \
--D5_layer_name             D5_LEAKYRELU        \
--D5_input_layer            4                   \
--D6                        cl_replay.api.layer.keras.Dropout_Layer \
--D6_rate                   0.3                 \
--D6_layer_name             D6_DROPOUT          \
--D6_input_layer            5                   \
--D7                        cl_replay.api.layer.keras.Dense_Layer \
--D7_layer_name             D7_DENSE            \
--D7_units                  1                   \
--D7_activation             none                \
--D7_input_layer            6                   \
--S_model_inputs            0                       \
--S_model_outputs           5                       \
--S0                        Input_Layer \
--S0_layer_name             S0_INPUT                \
--S0_shape                  28 28 1                 \
--S1                        cl_replay.api.layer.keras.Flatten_Layer \
--S1_layer_name             S1_FLATTEN              \
--S1_input_layer            0                       \
--S2                        cl_replay.api.layer.keras.Dense_Layer \
--S2_layer_name             S2_DENSE                \
--S2_units                  400                     \
--S2_activation             relu                    \
--S2_input_layer            1                       \
--S3                        cl_replay.api.layer.keras.Dense_Layer \
--S3_layer_name             S3_DENSE                \
--S3_units                  400                     \
--S3_activation             relu                    \
--S3_input_layer            2                       \
--S4                        cl_replay.api.layer.keras.Dense_Layer \
--S4_layer_name             S4_DENSE                \
--S4_units                  400                     \
--S4_activation             relu                    \
--S4_input_layer            3                       \
--S5                        cl_replay.api.layer.keras.Dense_Layer \
--S5_layer_name             S5_OUT                  \
--S5_units                  ${LABEL_DIM}            \
--S5_activation             softmax                 \
--S5_input_layer            4