export OPENAI_API_KEY=?
model_dir='lmsys/vicuna-13b-v1.3'
MODEL_NAME='vicuna'
#model_dir='WizardLMTeam/WizardLM-13B-V1.2'
#MODEL_NAME='wizardlm'
export TRANSFORMERS_CACHE=./

datasets=(samsum)

for i in ${datasets[@]}; do
    python experiments/run_ACING_summarization.py \
    --task $i \
    --n_prompt_tokens 5 \
    --total_iter 165 \
    --local_training_iter 1000 \
    --seed 0 \
    --intrinsic_dim 10 \
    --HF_cache_dir ${model_dir} \
    --gpt gpt-3.5-turbo \
    --name iter165_gpt-0301
done