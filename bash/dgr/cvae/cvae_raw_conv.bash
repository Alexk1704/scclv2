BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="dgr-cvae-mnist"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
LATENT_DIM=100
DATA_DIM=784
LABEL_DIM=10
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.dgr.experiment.Experiment_DGR \
--exp_id                        ${EXP_ID}                       \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_name                  mnist                           \
--dataset_load                  tfds                            \
--vis_gen                       no                              \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     6                               \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4                       \
--T2                            5                               \
--T3                            6                               \
--T4                            7                               \
--T5                            8                               \
--T6                            9                               \
--num_classes                   10                              \
--batch_size                    100                             \
--sampling_batch_size           100                             \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    DGR-VAE                         \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR}"                   \
--vis_path                      "${SAVE_DIR}/vis"               \
--ckpt_dir                      "${SAVE_DIR}"                   \
--log_training                  no                              \
--input_size                    28 28 1                         \
--generator_type                VAE                             \
--vae_epochs                    100                             \
--solver_epochs                 50                              \
--solver_epsilon                0.001                           \
--replay_proportions            -1. -1.                         \
--samples_to_generate           1.                              \
--loss_coef                     off                             \
--latent_dim                    ${LATENT_DIM}                   \
--vae_beta                      1.                              \
--enc_cond_input                yes                             \
--dec_cond_input                yes                             \
--drop_solver                   no                              \
--drop_generator                no                              \
--adam_epsilon                  0.001                           \
--adam_beta1                    0.9                             \
--adam_beta2                    0.999                           \
--vae_epsilon                   0.0001                          \
--E_model_inputs            0 1             \
--E_model_outputs           10 11           \
--E0                        Input_Layer \
--E0_layer_name             E0_INPUT        \
--E0_shape                  28 28 1         \
--E1                        Input_Layer \
--E1_layer_name             E1_INPUT        \
--E1_shape                  ${LABEL_DIM}    \
--E2                        cl_replay.api.layer.keras.Dense_Layer \
--E2_layer_name             E2_DENSE        \
--E2_units                  784             \
--E2_activation             relu            \
--E2_input_layer            1               \
--E3                        cl_replay.api.layer.keras.Reshape_Layer \
--E3_layer_name             E3_RESHAPE      \
--E3_target_shape           28 28 1         \
--E3_input_layer            2               \
--E4                        cl_replay.api.layer.keras.Concatenate_Layer \
--E4_layer_name             E4_CONCAT       \
--E4_input_layer            0 3             \
--E5                        cl_replay.api.layer.keras.Conv2D_Layer \
--E5_layer_name             E5_CONV2D       \
--E5_filters                32              \
--E5_kernel_size            3 3             \
--E5_strides                2 2             \
--E5_activation             relu            \
--E5_input_layer            4               \
--E6                        cl_replay.api.layer.keras.Conv2D_Layer \
--E6_layer_name             E6_CONV2D       \
--E6_filters                64              \
--E6_kernel_size            3 3             \
--E6_strides                2 2             \
--E6_activation             relu            \
--E6_input_layer            5               \
--E7                        cl_replay.api.layer.keras.Flatten_Layer \
--E7_layer_name             E7_FLATTEN      \
--E7_input_layer            6               \
--E8                        cl_replay.api.layer.keras.Dense_Layer \
--E8_layer_name             E8_DENSE        \
--E8_units                  512             \
--E8_activation             relu            \
--E8_input_layer            7               \
--E9                        cl_replay.api.layer.keras.Dense_Layer \
--E9_layer_name             E9_DENSE        \
--E9_units                  256             \
--E9_activation             relu            \
--E9_input_layer            8               \
--E10                       cl_replay.api.layer.keras.Dense_Layer \
--E10_layer_name            E10_MEAN        \
--E10_units                 ${LATENT_DIM}   \
--E10_activation            none            \
--E10_input_layer           9               \
--E11                       cl_replay.api.layer.keras.Dense_Layer \
--E11_layer_name            E11_LOGVAR      \
--E11_units                 ${LATENT_DIM}   \
--E11_activation            none            \
--E11_input_layer           9               \
--D_model_inputs            0 1                 \
--D_model_outputs           9                   \
--D0                        Input_Layer \
--D0_layer_name             D0_INPUT            \
--D0_shape                  ${LATENT_DIM}       \
--D1                        Input_Layer \
--D1_layer_name             D1_INPUT            \
--D1_shape                  ${LABEL_DIM}        \
--D2                        cl_replay.api.layer.keras.Dense_Layer \
--D2_layer_name             D2_DENSE        \
--D2_units                  ${LATENT_DIM}   \
--D2_activation             relu            \
--D2_input_layer            1               \
--D3                        cl_replay.api.layer.keras.Concatenate_Layer \
--D3_layer_name             D3_CONCAT           \
--D3_input_layer            0 2                 \
--D4                        cl_replay.api.layer.keras.Dense_Layer \
--D4_layer_name             D4_DENSE        \
--D4_units                  256             \
--D4_activation             relu            \
--D4_input_layer            3               \
--D5                        cl_replay.api.layer.keras.Dense_Layer \
--D5_layer_name             D5_DENSE        \
--D5_units                  512             \
--D5_activation             relu            \
--D5_input_layer            4               \
--D6                        cl_replay.api.layer.keras.Dense_Layer \
--D6_layer_name             D6_DENSE            \
--D6_units                  3136                \
--D6_activation             relu                \
--D6_input_layer            5                   \
--D7                        cl_replay.api.layer.keras.Reshape_Layer \
--D7_layer_name             D7_RESHAPE          \
--D7_target_shape           7 7 64              \
--D7_input_layer            6                   \
--D8                        cl_replay.api.layer.keras.Deconv2D_Layer \
--D8_layer_name             D8_DECONV2D         \
--D8_filters                32                  \
--D8_kernel_size            3 3                 \
--D8_strides                2 2                 \
--D8_activation             relu                \
--D8_input_layer            7                   \
--D9                        cl_replay.api.layer.keras.Deconv2D_Layer \
--D9_layer_name             D9_DECONV2D         \
--D9_filters                1                   \
--D9_kernel_size            3 3                 \
--D9_strides                2 2                 \
--D9_activation             none                \
--D9_input_layer            8                   \
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