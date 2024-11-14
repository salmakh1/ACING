# import re
# import numpy as np
# import glob
#
#
# def extract_info(log_file):
#     # Variables to store important information
#     final_test_score = None
#     best_instruction = None
#     best_reward = None
#
#     # Regular expressions for matching the important info
#     score_pattern = r'the test score is \s*(\d+\.\d+|\d+)'  # Matches final test score
#     instruction_pattern = r'the best prompt is \s*(.*)'  # Matches the best instruction
#     reward_pattern = r'the prompt achieved reward equals to [\s*(.*)'  # Matches the best instruction
#
#     # Read the log file line by line
#     with open(log_file, 'r', encoding='utf-8') as file:
#         for line in file:
#             # Search for the final test score
#             score_match = re.search(score_pattern, line)
#             if score_match:
#                 final_test_score = float(score_match.group(1))
#
#             # Search for the best instruction
#             instruction_match = re.search(instruction_pattern, line)
#             if instruction_match:
#                 best_instruction = instruction_match.group(1).strip()
#
#             # Search for the best reward
#             # reward_match = re.search(reward_pattern, line)
#             # if reward_match:
#             #     best_reward = reward_match.group(1).strip()
#     print(final_test_score)
#     print(best_instruction)
#     # print(best_reward)
#
#     return final_test_score, best_instruction
#     # , best_reward
#
#
# # tasks = [
# #     "active_to_passive", "antonyms", "auto_categorization", "auto_debugging",
# #     "cause_and_effect", "common_concept", "diff", "first_word_letter",
# #     "informal_to_formal", "larger_animal", "letters_list", "negation",
# #     "num_to_verbal", "odd_one_out", "object_counting", "orthography_starts_with",
# #     "periodic_elements", "rhymes", "second_word_letter", "sentence_similarity",
# #     "sentiment", "singular_to_plural", "sum", "synonyms", "taxonomy_animal",
# #     "translation_en-de", "translation_en-es", "translation_en-fr",
# #     "word_sorting", "word_unscrambling"
# # ]
#
#
# tasks = [
#     "antonyms", "auto_categorization", "auto_debugging",
#     "cause_and_effect", "common_concept",
#     "informal_to_formal", "negation",
#     "odd_one_out", "object_counting", "orthography_starts_with",
#     "rhymes", "second_word_letter", "sentence_similarity",
#     "sum", "synonyms", "taxonomy_animal",
#     "word_sorting", "word_unscrambling"
# ]
#
# # Open results.txt for writing
# with open('results.txt', 'w') as file:
#     file.write(f"For the method OAC we have:\n")
#     all_tests = []
#     all_rewards = []
#
#     for task in tasks:
#         # Path pattern to match files with different seeds
#         base_dir = r'OAC_512_task_'
#         log_file_pattern = base_dir + task + '_seed_*'
#         # Loop through all files matching the pattern
#         rewards = []
#         tests = []
#         prompts = []
#
#         max_test = -1
#         for log_file_path in glob.glob(log_file_pattern):
#             print("hereeee")
#             print(log_file_path)
#             final_score, best_inst = extract_info(log_file_path)
#
#             tests.append(float(final_score))
#
#             if float(final_score) > max_test:
#                 prompts.append(best_inst)
#                 max_test = float(final_score)
#         #
#         # average_reward = np.mean(np.array(rewards))
#         # deviation_reward = np.std(np.array(rewards))
#         # median_reward = np.median(np.array(rewards))
#
#         average_test = np.mean(np.array(tests))
#         deviation_test = np.std(np.array(tests))
#         median_test = np.median(np.array(tests))
#
#         all_tests.append(average_test)
#         # all_rewards.append(average_reward)
#
#         file.write("\n")
#         file.write(f"{task} | t {average_test:.2f} ({deviation_test:.2f})  | {prompts[-1][2:-2]} \n")
#
#         # file.write(f"{task} | {average_test:.2f} ({deviation_test:.2f}) | {prompts[-1]} \n")
#
#         print("\n")
#         print(f"For the task {task} we have:")
#         print(f"Best Instruction on Test: {prompts[-1]}")
#         print(f"Final Test Score is: {average_test:.2f} +- ({deviation_test:.2f}) with median ({median_test:.2f})")
#
#     file.write("\n")
#     file.write(
#         f"OAC Test {np.mean(all_tests):.2f} +- ({np.std(all_tests):.2f}) with median ({np.median(all_tests):.2f}) \n")






import re
import numpy as np
import glob
import csv

def extract_info(log_file):
    # Variables to store important information
    final_test_score = None
    best_instruction = None

    # Regular expressions for matching the important info
    score_pattern = r'the test score is \s*(.*)'  # Matches final test score
    instruction_pattern = r'the best prompt is \s*(.*)'  # Matches the best instruction

    # Read the log file line by line
    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Search for the final test score
            score_match = re.search(score_pattern, line)
            if score_match:
                final_test_score = float(score_match.group(1))

            # Search for the best instruction
            instruction_match = re.search(instruction_pattern, line)
            if instruction_match:
                best_instruction = instruction_match.group(1).strip()

    return final_test_score, best_instruction

tasks = [
    "active_to_passive", "antonyms", "auto_categorization", "auto_debugging",
    "cause_and_effect", "common_concept", "diff", "first_word_letter",
    "informal_to_formal", "larger_animal", "letters_list", "negation",
    "num_to_verbal", "odd_one_out", "object_counting", "orthography_starts_with",
    "periodic_elements", "rhymes", "second_word_letter", "sentence_similarity",
    "sentiment", "singular_to_plural", "sum", "synonyms", "taxonomy_animal",
    "translation_en-de", "translation_en-es", "translation_en-fr",
    "word_sorting", "word_unscrambling"
]
# tasks = [
#     "active_to_passive",  "larger_animal",
#     "num_to_verbal",
#     "periodic_elements",  "singular_to_plural",
#     "translation_en-de", "translation_en-es", "translation_en-fr",
# ]
# Create and open the CSV file for writing
with open('results_sac_d_10_seed_0.csv', 'w', newline='') as csvfile:
    # Define the fieldnames (columns) in the CSV
    fieldnames = ['Task', 'Average Test', 'Standard Deviation', 'Median Test', 'Best Prompt']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header row
    writer.writeheader()

    all_tests = []

    for task in tasks:
        # Path pattern to match files with different seeds
        base_dir = r'SAC_d_10/SAC_d_10_task_'
        # base_dir = r'ape/ape_prime_task_'
        log_file_pattern = base_dir + task + '_seed_0*'

        # Loop through all files matching the pattern
        tests = []
        prompts = []

        max_test = -1
        for log_file_path in glob.glob(log_file_pattern):
            print(log_file_path)
            final_score, best_inst = extract_info(log_file_path)
            print(final_score)
            tests.append(float(final_score))

            if float(final_score) > max_test:
                prompts.append(best_inst)
                max_test = float(final_score)

        average_test = np.mean(np.array(tests))
        deviation_test = np.std(np.array(tests))
        median_test = np.median(np.array(tests))

        all_tests.append(average_test)

        # Write task results to CSV
        writer.writerow({
            'Task': task,
            'Average Test': f"{average_test:.2f}",
            'Standard Deviation': f"{deviation_test:.2f}",
            'Median Test': f"{median_test:.2f}",
            'Best Prompt': prompts[-1][2:-2]  # Assuming you want to remove some characters
        })

    # Optionally, print the summary results
    print(f"OAC Test {np.mean(all_tests):.2f} +- ({np.std(all_tests):.2f}) with median ({np.median(all_tests):.2f})")











# for task in tasks:
#
#
#
#     # Loop through all files matching the pattern
#     rewards = []
#     tests = []
#     prompts = []
#
#     max_test = -1
#     for log_file_path in glob.glob(log_file_pattern):
#         final_score, best_inst, best_reward = extract_info(log_file_path)
#
#         rewards.append(float(best_reward))
#         tests.append(float(final_score))
#
#         if float(final_score) > max_test:
#             prompts.append(best_inst)
#             max_test = float(final_score)
#
#     average_reward = np.mean(np.array(rewards))
#     deviation_reward = np.std(np.array(rewards))
#     median_reward = np.median(np.array(rewards))
#
#     average_test = np.mean(np.array(tests))
#     deviation_test = np.std(np.array(tests))
#     median_test = np.median(np.array(tests))

# print("\n")
# print(f"For the task {task} we have:")
# print(f"Best Instruction on Test: {prompts[-1]}")
# print(f"Final Test Score is: {average_test:.2f} +- ({deviation_test:.2f}) with median ({median_test:.2f})")
# print(f"Best Reward: {average_reward:.2f} +- ({deviation_reward:.2f}) with median ({median_reward:.2f})")
