BASE=$HOME
export PYTHONPATH=$PYTHONPATH:${BASE}/git/scclv2/src
EXP_ID="sa-mnist-rts"
SAVE_DIR="${BASE}/exp-results/${EXP_ID}"
DATA_DIR="${BASE}/custom_datasets/"
AR_MOD="cl_replay.architecture.ar"
LATENT_DIM=50
DATA_DIM=784
LABEL_DIM=10
# export INVOCATION="singularity exec --nv  ${BASE}/ubuntu24tf217.sif"
export INVOCATION=""
${INVOCATION} python3 -m cl_replay.architecture.dgr.experiment.Experiment_SA \
--exp_id                        "${EXP_ID}"                     \
--log_level                     DEBUG                           \
--random_seed                   42                              \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_load                  tfds                            \
--dataset_name                  fashion_mnist                   \
--vis_batch                     no                              \
--vis_gen                       no                              \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     7                               \
--forgetting_mode               separate                        \
--forgetting_tasks              2 4 6                           \
--DAll                          0 1 2 3 4 5 6 7 8 9             \
--T1                            0 1 2 3 4 5                     \
--T2                            0 1 2                           \
--T3                            6 7                             \
--T4                            3 4                             \
--T5                            8                               \
--T6                            5                               \
--T7                            9                               \
--extra_eval                    6 7 8 9                         \
--num_classes                   10                              \
--batch_size                    128                             \
--sampling_batch_size           128                             \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    DGR-VAE                         \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR:?}"                 \
--vis_path                      "${SAVE_DIR:?}/vis"             \
--ckpt_dir                      "${SAVE_DIR:?}"                 \
--log_training                  no                              \
--input_size                    1 1 2048                        \
--generator_type                VAE                             \
--vae_epochs                    100                             \
--solver_epochs                 100                             \
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
--adam_epsilon                  0.0001                          \
--adam_beta1                    0.9                             \
--adam_beta2                    0.999                           \
--vae_epsilon                   0.0001                          \
--amnesiac                      yes                             \
--sa_forg_iters                 10000                           \
--sa_fim_samples                10000                           \
--sa_lambda                     100.                            \
--sa_gamma                      1.0                             \
--E_model_inputs            0 2             \
--E_model_outputs           7 8             \
--E0                        Input_Layer \
--E0_layer_name             E0_INPUT        \
--E0_shape                  28 28 1         \
--E1                        cl_replay.api.layer.keras.Flatten_Layer \
--E1_layer_name             E1_FLATTEN      \
--E1_input_layer            0               \
--E2                        Input_Layer \
--E2_layer_name             E2_INPUT        \
--E2_shape                  ${LABEL_DIM}    \
--E3                        cl_replay.api.layer.keras.Concatenate_Layer \
--E3_layer_name             E3_CONCAT       \
--E3_input_layer            1 2             \
--E4                        cl_replay.api.layer.keras.Dense_Layer \
--E4_layer_name             E4_DENSE        \
--E4_units                  512             \
--E4_activation             relu            \
--E4_input_layer            3               \
--E5                        cl_replay.api.layer.keras.Dense_Layer \
--E5_layer_name             E5_DENSE        \
--E5_units                  256             \
--E5_activation             relu            \
--E5_input_layer            4               \
--E6                        cl_replay.api.layer.keras.Dense_Layer \
--E6_layer_name             E6_DENSE        \
--E6_units                  128             \
--E6_activation             relu            \
--E6_input_layer            5               \
--E7                        cl_replay.api.layer.keras.Dense_Layer \
--E7_layer_name             E7_MEAN         \
--E7_units                  ${LATENT_DIM}   \
--E7_activation             none            \
--E7_input_layer            6               \
--E8                        cl_replay.api.layer.keras.Dense_Layer \
--E8_layer_name             E8_LOGVAR       \
--E8_units                  ${LATENT_DIM}   \
--E8_activation             none            \
--E8_input_layer            6               \
--D_model_inputs            0 1                 \
--D_model_outputs           7                   \
--D0                        Input_Layer \
--D0_layer_name             D0_INPUT            \
--D0_shape                  ${LATENT_DIM}       \
--D1                        Input_Layer \
--D1_layer_name             D1_INPUT            \
--D1_shape                  ${LABEL_DIM}        \
--D2                        cl_replay.api.layer.keras.Concatenate_Layer \
--D2_layer_name             D2_CONCAT           \
--D2_input_layer            0 1                 \
--D3                        cl_replay.api.layer.keras.Dense_Layer \
--D3_layer_name             D3_DENSE            \
--D3_units                  128                 \
--D3_activation             relu                \
--D3_input_layer            2                   \
--D4                        cl_replay.api.layer.keras.Dense_Layer \
--D4_layer_name             D4_DENSE            \
--D4_units                  256                 \
--D4_activation             relu                \
--D4_input_layer            3                   \
--D5                        cl_replay.api.layer.keras.Dense_Layer \
--D5_layer_name             D5_DENSE            \
--D5_units                  512                 \
--D5_activation             relu                \
--D5_input_layer            4                   \
--D6                        cl_replay.api.layer.keras.Dense_Layer \
--D6_layer_name             D6_DENSE            \
--D6_units                  ${DATA_DIM}         \
--D6_activation             none                \
--D6_input_layer            5                   \
--D7                        cl_replay.api.layer.keras.Reshape_Layer \
--D7_layer_name             D7_RESHAPE          \
--D7_target_shape           28 28 1             \
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