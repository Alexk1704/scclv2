BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/scclv2/src
EXP_ID="dgr-vae-features"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
LATENT_DIM=100
DATA_DIM=2048
LABEL_DIM=10
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.dgr.experiment.Experiment_DGR \
--exp_id                        ${EXP_ID}                       \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_name                  svhn-7-ex.npz                   \
--dataset_load                  from_npz                        \
--vis_gen                       no                              \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     4                               \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4 5 6                   \
--T2                            7                               \
--T3                            8                               \
--T4                            9                               \
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
--input_size                    1 1 2048                        \
--generator_type                VAE                             \
--vae_epochs                    200                             \
--solver_epochs                 50                              \
--solver_epsilon                0.001                           \
--replay_proportions            -1. -1.                         \
--samples_to_generate           1.                              \
--loss_coef                     off                             \
--recon_loss                    binary_crossentropy             \
--latent_dim                    ${LATENT_DIM}                   \
--vae_beta                      1.                              \
--enc_cond_input                no                              \
--dec_cond_input                no                              \
--drop_solver                   no                              \
--drop_generator                no                              \
--adam_epsilon                  0.001                           \
--adam_beta1                    0.9                             \
--adam_beta2                    0.999                           \
--vae_epsilon                   0.00001                         \
--E_model_inputs            0               \
--E_model_outputs           5 6             \
--E0                        Input_Layer \
--E0_layer_name             E0_INPUT        \
--E0_shape                  1 1 2048        \
--E1                        cl_replay.api.layer.keras.Flatten_Layer \
--E1_layer_name             E1_FLATTEN      \
--E1_input_layer            0               \
--E2                        cl_replay.api.layer.keras.Dense_Layer \
--E2_layer_name             E2_DENSE        \
--E2_units                  2048            \
--E2_activation             relu            \
--E2_input_layer            1               \
--E3                        cl_replay.api.layer.keras.Dense_Layer \
--E3_layer_name             E3_DENSE        \
--E3_units                  2048            \
--E3_activation             relu            \
--E3_input_layer            2               \
--E4                        cl_replay.api.layer.keras.Dense_Layer \
--E4_layer_name             E4_DENSE        \
--E4_units                  2048            \
--E4_activation             relu            \
--E4_input_layer            3               \
--E5                        cl_replay.api.layer.keras.Dense_Layer \
--E5_layer_name             E5_MU           \
--E5_units                  ${LATENT_DIM}   \
--E5_activation             none            \
--E5_input_layer            4               \
--E6                        cl_replay.api.layer.keras.Dense_Layer \
--E6_layer_name             E6_RHO          \
--E6_units                  ${LATENT_DIM}   \
--E6_activation             none            \
--E6_input_layer            4               \
--D_model_inputs            0                   \
--D_model_outputs           5                   \
--D0                        Input_Layer \
--D0_layer_name             D0_INPUT            \
--D0_shape                  ${LATENT_DIM}       \
--D1                        cl_replay.api.layer.keras.Dense_Layer \
--D1_layer_name             D1_DENSE        \
--D1_units                  2048            \
--D1_activation             relu            \
--D1_input_layer            0               \
--D2                        cl_replay.api.layer.keras.Dense_Layer \
--D2_layer_name             D2_DENSE        \
--D2_units                  2048            \
--D2_activation             relu            \
--D2_input_layer            1               \
--D3                        cl_replay.api.layer.keras.Dense_Layer \
--D3_layer_name             D3_DENSE        \
--D3_units                  2048            \
--D3_activation             relu            \
--D3_input_layer            2               \
--D4                        cl_replay.api.layer.keras.Dense_Layer \
--D4_layer_name             D4_DENSE            \
--D4_units                  ${DATA_DIM}         \
--D4_activation             sigmoid             \
--D4_input_layer            3                   \
--D5                        cl_replay.api.layer.keras.Reshape_Layer \
--D5_layer_name             D5_RESHAPE          \
--D5_target_shape           1 1 2048            \
--D5_input_layer            4                   \
--S_model_inputs            0                       \
--S_model_outputs           4                       \
--S0                        Input_Layer \
--S0_layer_name             S0_INPUT                \
--S0_shape                  1 1 2048                \
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
--S4_layer_name             S3_OUT                  \
--S4_units                  ${LABEL_DIM}            \
--S4_activation             softmax                 \
--S4_input_layer            3