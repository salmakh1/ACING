export OPENAI_API_KEY=?
model_dir='lmsys/vicuna-13b-v1.3'
MODEL_NAME='vicuna'
#model_dir='WizardLMTeam/WizardLM-13B-V1.2'
#MODEL_NAME='wizardlm'
export TRANSFORMERS_CACHE=./

datasets=(active_to_passive antonyms auto_categorization auto_debugging cause_and_effect common_concept diff first_word_letter informal_to_formal larger_animal letters_list negation num_to_verbal odd_one_out object_counting orthography_starts_with periodic_elements rhymes second_word_letter sentence_similarity sentiment singular_to_plural sum synonyms taxonomy_animal translation_en-de translation_en-es translation_en-fr word_sorting word_unscrambling)


for i in ${datasets[@]}; do
    python experiments/run_ACING.py \
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