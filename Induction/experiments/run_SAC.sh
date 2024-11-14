export OPENAI_API_KEY=?
model_dir='lmsys/vicuna-13b-v1.3'
MODEL_NAME='vicuna'
#model_dir='WizardLMTeam/WizardLM-13B-V1.2'
#MODEL_NAME='wizardlm'
export TRANSFORMERS_CACHE=./
#antonyms common_concept diff first_word_letter informal_to_formal letters_list

#
#datasets=(taxonomy_animal diff letters_list antonyms second_word_letter cause_and_effect rhymes synonyms informal_to_formal word_unscrambling auto_categorization auto_debugging common_concept negation odd_one_out object_counting orthography_starts_with sentence_similarity sum word_sorting active_to_passive larger_animal num_to_verbal periodic_elements singular_to_plural translation_en-de translation_en-es translation_en-fr)

#datasets=(active_to_passive antonyms auto_categorization auto_debugging cause_and_effect common_concept diff first_word_letter informal_to_formal larger_animal letters_list negation num_to_verbal odd_one_out object_counting orthography_starts_with periodic_elements rhymes second_word_letter sentence_similarity sentiment singular_to_plural sum synonyms taxonomy_animal translation_en-de translation_en-es translation_en-fr word_sorting word_unscrambling)

datasets=(word_unscrambling)

#datasets=(orthography_starts_with periodic_elements rhymes second_word_letter sentence_similarity sentiment singular_to_plural sum synonyms taxonomy_animal translation_en-de translation_en-es translation_en-fr word_sorting word_unscrambling)


#datasets=(samsum)
#
#second_word_letter antonyms auto_categorization auto_debugging cause_and_effect common_concept diff first_word_letter informal_to_formal larger_animal letters_list negation odd_one_out object_counting orthography_starts_with rhymes sentence_similarity sentiment sum synonyms taxonomy_animal word_sorting word_unscrambling

#object_counting odd_one_out sentence_similarity

#(second_word_letter cause_and_effect rhymes synonyms informal_to_formal word_unscrambling auto_categorization auto_debugging common_concept negation odd_one_out object_counting orthography_starts_with sentence_similarity sum taxonomy_animal word_sorting antonyms)

#antonyms cause_and_effect synonyms informal_to_formal rhymes
#second_word_letter
#active_to_passive antonyms auto_categorization auto_debugging cause_and_effect common_concept diff first_word_letter informal_to_formal larger_animal letters_list negation num_to_verbal odd_one_out object_counting orthography_starts_with periodic_elements rhymes second_word_letter sentence_similarity sentiment singular_to_plural sum synonyms taxonomy_animal translation_en-de translation_en-es translation_en-fr word_sorting word_unscrambling)
#datasets=(second_word_letter)


for i in ${datasets[@]}; do
    python experiments/run_SAC_bandits.py \
    --task $i \
    --n_prompt_tokens 5 \
    --total_iter 165 \
    --local_training_iter 1000 \
    --seed 2 \
    --intrinsic_dim 10 \
    --HF_cache_dir ${model_dir} \
    --gpt gpt-3.5-turbo \
    --name iter165_gpt-0301
done