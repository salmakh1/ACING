import logging
import random
import time

import torch
import numpy as np
import copy
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd)
from experiments.SAC import SAC

from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
from transformers import AutoTokenizer, LlamaForCausalLM, BertForSequenceClassification
from LlamaForMLPRegression import LlamaForMLPRegression, MLPRegression, MLPRegression_Train, NeuralTSDiag
from automatic_prompt_engineer import evaluate, config, template, data
from experiments.SAC import SACAgent

import os
import re

from tqdm import tqdm
import argparse
from experiments.evaluation.instruction_induction.utility import set_all_seed, TASKS

import matplotlib.pyplot as plt

SMOKE_TEST = os.environ.get("SMOKE_TEST")
## bayesian opt
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


model_name = "vicuna"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_model = 'chatgpt'
alpha = 1
sigma = 1


class LMForwardAPI:
    def __init__(self, model_name='vicuna', eval_data=None, init_prompt=None, init_qa=None, conf=None, base_conf=None,
                 prompt_gen_data=None, n_prompt_tokens=None, few_shot_data=None,
                 HF_cache_dir=None, random_proj=None, intrinsic_dim=None):
        print("HEEEEREEEEEE")
        p = torch.ones(10)

        torch.cuda.empty_cache()

        kwargs = {'torch_dtype': torch.float16}
        if model_name in ["vicuna", "alpaca", "flan-t5", "wizardlm"]:
            self.model = LlamaForMLPRegression.from_pretrained(
                HF_cache_dir, low_cpu_mem_usage=True, **kwargs
            ).cuda()

            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_cache_dir,
                model_max_length=512,
                padding_side='left',
                use_fast=False,
            )

        else:
            raise NotImplementedError

        self.exemplars = init_qa[0]
        self.init_token = init_prompt[0] + init_qa[0]
        print("init_prompt", init_prompt)
        print("init_qa[0]", init_qa[0])

        if model_name in ['alpaca', 'vicuna', "wizardlm"]:
            self.embedding = self.model.get_input_embeddings().weight.detach()
            input_ids = self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()
            # print("input_ids", input_ids)
            self.init_prompt = self.embedding[input_ids]

        ################# setup n_prompts_token #################
        self.n_prompt_tokens = n_prompt_tokens
        self.hidden_size = self.init_prompt.shape[-1]
        # print("self.hidden_size", self.hidden_size)
        print('Shape of initial prompt embedding: {}'.format(self.init_prompt.shape))

        # self.init_prompt = self.init_prompt.reshape(self.n_prompt_tokens * self.hidden_size)
        self.count = 0
        print("dim is ", self.n_prompt_tokens * self.hidden_size)
        self.linear = torch.nn.Linear(intrinsic_dim, self.n_prompt_tokens * self.hidden_size, bias=False)  #

        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['llama', 'alpaca', 'vicuna', "wizardlm"]:
                print('Get the embedding firstly to avoid issues')
            else:
                raise NotImplementedError
            mu_hat = np.mean(self.embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(self.embedding.reshape(-1).detach().cpu().numpy())
            mu = 0.0
            std = alpha * std_hat / (np.sqrt(intrinsic_dim) * sigma)

            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.uniform_(p, -1, 1)
        elif random_proj == 'uniform':
            for p in self.linear.parameters():
                torch.nn.init.uniform_(p, -1, 1)

        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        # print("eval_data is ", self.eval_data)
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        if api_model in ['llama', 'flan-t5']:
            self.api_model = exec_evaluator(api_model, self.conf)

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.prompts_list = []
        self.rewards = {}

    def get_last_token_hidden_state(self, prompt_embedding):

        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding_ = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype).reshape(1,
                                                                                                            self.n_prompt_tokens,
                                                                                                            -1)
        print("shape emb", prompt_embedding_.shape)
        print("shape input emb", input_embed.shape)

        input_embed = torch.cat((prompt_embedding_, input_embed), 1)
        last_token_id = input_embed.shape[1] - 1
        # last_token_id = 0
        hidden_state, = self.model.get_last_token_hidden_state(inputs_embeds=input_embed,
                                                               sequence_lengths=last_token_id)
        return hidden_state

    def get_last_token_hidden_state_batch(self, prompt_embedding, pooling='last'):
        size = prompt_embedding.shape[0]
        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()

        batch_size = 10
        n_batchs = size // batch_size + int((size % batch_size) != 0)
        all_hidden_state = []
        for i in tqdm(range(n_batchs), desc='Get hidden states'):
            if i == n_batchs - 1:
                prompt_batch = prompt_embedding[(i * batch_size):]
            else:
                prompt_batch = prompt_embedding[(i * batch_size):((i + 1) * batch_size)]
            batch_size_ = prompt_batch.shape[0]
            input_embed = self.embedding[input_ids]
            print('Size of the input is: {}'.format(input_embed.shape))
            input_embed = input_embed.repeat(batch_size_, 1, 1)
            prompt_embedding_ = prompt_batch.to(device=input_embed.device, dtype=input_embed.dtype).reshape(batch_size_,
                                                                                                            self.n_prompt_tokens,
                                                                                                            -1)
            input_embed = torch.cat((prompt_embedding_, input_embed), 1)
            last_token_id = input_embed.shape[1] - 1

            hidden_state_, = self.model.get_last_token_hidden_state(inputs_embeds=input_embed,
                                                                    sequence_lengths=last_token_id, pooling=pooling)
            all_hidden_state.append(hidden_state_)

        all_hidden_state = torch.vstack(all_hidden_state)
        return all_hidden_state

    def evaluate_last(self, instruction, previous_r, n_evaluations=3):
        self.num_call += 1
        b = False

        # postprocess instruction
        instruction[0] = instruction[0]
        start = instruction[0].find('The instruction was to')
        end = instruction[0].find('Comment:')
        if end == -1:
            instruction[0] = instruction[0][start:]
        else:
            instruction[0] = instruction[0][start: end]

        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        search_string = 'The instruction was to'
        for sentence in sentences:
            if sentence.startswith(search_string):
                instruction[0] = sentence
                break

        # print post-processed instruction
        print('Instruction: {}'.format(instruction))
        print("The previous reward is ", previous_r)
        if api_model in ['chatgpt']:
            print("previous_r", previous_r)
            # rewards = [previous_r]
            if isinstance(previous_r, list):
                rewards = previous_r
            else:
                rewards = [previous_r]
            print("####Rewards is ", rewards)
            for i in range(n_evaluations):
                dev_perf, instruction_score = evaluate.evaluate_prompts(instruction, self.eval_template,
                                                                        self.eval_data,
                                                                        self.demos_template, self.few_shot_data,
                                                                        self.conf['evaluation']['method'],
                                                                        self.conf['evaluation'])

                dev_perf = dev_perf.sorted()[1][0]
                print(dev_perf)
                rewards.append(dev_perf)
            print(rewards)
            r = np.mean(np.array(rewards))
            dev_perf = r

        else:
            raise NotImplementedError

        self.prompts_list.append((len(self.prompts_list), instruction[0], dev_perf))
        if dev_perf >= self.best_last_perf:
            self.count += 1

        # if dev_perf >= self.best_dev_perf:
        #     self.best_dev_perf = dev_perf
        #     self.best_prompt = copy.deepcopy(tmp_prompt)
        #     self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score, b

    def eval(self, prompt_embedding=None, test_data=None, n_evaluations=1):
        self.num_call += 1
        b = False
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = prompt_embedding.detach().clone()  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim

        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            # prompt_embedding = self.linear(prompt_embedding)  # Az
            # if self.init_prompt is not None:
            #     prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)

        elif isinstance(prompt_embedding, torch.Tensor):
            prompt_embedding = prompt_embedding.type(torch.float32)
            # prompt_embedding = self.linear(prompt_embedding)  # Az
            prompt_embedding = prompt_embedding.reshape(1, self.n_prompt_tokens, -1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )

        print("self.init_token ", self.init_token)
        input_ids = self.tokenizer(self.init_token, return_tensors="pt").input_ids.cuda()
        input_embed = self.embedding[input_ids]
        prompt_embedding = prompt_embedding.to(device=input_embed.device, dtype=input_embed.dtype)

        input_embed = torch.cat((prompt_embedding, input_embed), 1)
        print(input_embed.shape)
        outputs = self.model.generate(inputs_embeds=input_embed, max_new_tokens=64)
        instruction = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        instruction[0] = 'The instruction was to ' + instruction[0]
        start = instruction[0].find('The instruction was to')
        end = instruction[0].find('Comment:')
        if end == -1:
            instruction[0] = instruction[0][start:]
        else:
            instruction[0] = instruction[0][start: end]

        sentences = re.split(r' *[\.\?!][\'"\)\]]* *', instruction[0])
        search_string = 'The instruction was to'
        for sentence in sentences:
            if sentence.startswith(search_string):
                instruction[0] = sentence
                break

        # print post-processed instruction
        print('Instruction: {}'.format(instruction))

        if instruction[0] in self.prompts_set.keys():
            print("prompt exists")
            (dev_perf, instruction_score) = self.prompts_set[instruction[0]]
            b = True
        else:
            if api_model in ['chatgpt']:
                print("calling the chatgpt function")
                # scores = []
                dev_perf, instruction_score = evaluate.evaluate_prompts(instruction, self.eval_template,
                                                                        self.eval_data,
                                                                        self.demos_template, self.few_shot_data,
                                                                        self.conf['evaluation']['method'],
                                                                        self.conf['evaluation'])

                dev_perf = dev_perf.sorted()[1][0]
                if instruction[0] not in self.rewards:
                    self.rewards[instruction[0]] = [dev_perf]
                self.prompts_set[instruction[0]] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        self.prompts_list.append((len(self.prompts_list), instruction[0], dev_perf))
        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_instruction = instruction

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score, b


    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set

    def return_prompts_list(self):
        return self.prompts_list


def plot(loss_values, file_name):
    # Example loss values (replace with your actual loss values)

    loss_values = np.array(loss_values)
    epochs = range(1, len(loss_values) + 1)  # Generate x-axis values

    # Plotting the loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel(file_name)
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(file_name, dpi=300)  # Save as a PNG file with 300 dpi
    plt.show()  # Display the plot


def run(task, n_prompt_tokens, HF_cache_dir, total_iter, random_proj,
        intrinsic_dim, gpt):

    # dataset = load_dataset("samsum",trust_remote_code=True)
    # print(dataset)
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())

    induce_data, test_data = load_data('induce', task), load_data('eval', task)
    # Get size of the induce data
    induce_data_size = len(induce_data[0])

    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    # Induce data is split into prompt_gen_data and eval_data
    prompt_gen_data, eval_data = data.create_split(
        induce_data, prompt_gen_size)

    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]

    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]"  # change the evaluation template
    init_prompt = ['\n']
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to"

    base_conf = '../experiments/configs/instruction_induction.yaml'
    print("base_conf is", base_conf)
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 5,
            'num_prompts_per_subsample': 20,
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        }
    }

    # make the demo automatically
    subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
    prompt_gen_template = template.InitQATemplate(prompt_gen_template)

    d_template = template.DemosTemplate(demos_template)
    demos = d_template.fill(subsampled_data)
    init_qa = [prompt_gen_template.fill(demos)]

    print(prompt_gen_data, HF_cache_dir, intrinsic_dim)

    model_forward_api = LMForwardAPI(model_name=model_name, eval_data=eval_data, init_prompt=init_prompt,
                                     init_qa=init_qa, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data,
                                     n_prompt_tokens=n_prompt_tokens, HF_cache_dir=HF_cache_dir,
                                     random_proj=random_proj, intrinsic_dim=intrinsic_dim)

    t = 0

    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': gpt
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
            'task': task,
            'num_samples': min(100, len(test_data[0])),
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                    'model': gpt
                }
            }
        }
    }
    #
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"SAC_d_{intrinsic_dim}_task_{task}_seed_{args.seed}.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    for handler in logger.handlers:
        handler.flush()

    logging.getLogger('openai').setLevel(logging.WARNING)  # Suppress verbose API logs

    logging.info(f"Start Training")

    logging.info("The exemplars are ")
    logging.info(model_forward_api.exemplars)

    active_arms = []
    rewards = {}
    values = []

    sac = SAC(dim=intrinsic_dim, bounds=np.array([[0, 1]] * intrinsic_dim), max_iter=total_iter,
              W_LLM=model_forward_api)

    bounds = np.array([[0, 1]] * intrinsic_dim)
    agent = SACAgent(dim=bounds.shape[0], llm=model_forward_api, seed=args.seed)
    state_tensor = torch.FloatTensor(torch.tensor([1.0])).unsqueeze(0)
    best_values = []
    while t < int(total_iter):
        print(f"We are in iteration {t}")
        if args.rep:
            if t == 150:
                print('Evaluate on test data...')
                prompts = model_forward_api.return_best_prompt()
                logging.info("Best instruction is:")
                logging.info(prompts)

                prompts_set = model_forward_api.return_prompts_set()
                logging.info("The final instruction set is:")
                logging.info(prompts_set)
                prompts_list = model_forward_api.return_prompts_list()
                logging.info("The final instruction list is:")
                logging.info(prompts_list)

                # ### check BEST prompts #########
                # #
                print("Check best instructions...")
                best_10_instuctions_to_reward = {}
                # Calculate the total reward for each instruction
                reward_sums = {key: np.mean(np.array(reward_list)) for key, reward_list in
                               model_forward_api.rewards.items()}
                # Sort the instructions by their total reward in descending order
                sorted_instructions = sorted(reward_sums.items(), key=lambda item: item[1], reverse=True)
                # Get the top 10 instructions with the highest rewards
                top_10_instructions = sorted_instructions[:5]
                print("top 5 instructions", top_10_instructions)
                # Extract the instruction keys and their rewards
                top_10_instruction_rewards = [(instruction, model_forward_api.rewards[instruction]) for instruction, _ in
                                              top_10_instructions]
                print("top_5_instruction_rewards", top_10_instruction_rewards)
                logging.info("top_5_instruction_rewards")
                logging.info(top_10_instruction_rewards)

                for instruction, reward in top_10_instruction_rewards:
                    r, _, b = model_forward_api.evaluate_last([instruction], reward)
                    best_10_instuctions_to_reward[instruction] = r
                    print(instruction, r)
                print(best_10_instuctions_to_reward)

                print('Evaluate on test data...')
                # prompts = model_forward_api.return_best_prompt()
                best_instruction = max(best_10_instuctions_to_reward, key=best_10_instuctions_to_reward.get)
                # If you also want to get the best reward value
                best_reward = best_10_instuctions_to_reward[best_instruction]
                print("best_reward", best_reward)
                logging.info("best_reward")
                logging.info(best_reward)
                prompts = [best_instruction]
                print("Best instruction is:")
                print(prompts)
                logging.info("Best instruction is:")
                logging.info(prompts)

                test_res = ape.evaluate_prompts(prompts=prompts,
                                                eval_template=eval_template,
                                                eval_data=test_data,
                                                few_shot_data=prompt_gen_data,
                                                demos_template=demos_template,
                                                conf=test_conf,
                                                base_conf=base_conf)
                test_res = test_res[0]
                test_score = test_res.sorted()[1][0]
                logging.info("##########################################################")
                logging.info(f"test score at iteration {t} after picking from top-5 is {test_score}")
                logging.info("##########################################################")

        # Select action
        action = agent.select_action(state_tensor)
        # action = agent.optimistic_sample(state_tensor)

        print("action", action)

        # Execute action in environment and get reward
        reward, b = sac.reward_function(action)
        count = 0
        while b:
            action = agent.select_action(state_tensor)
            print("action", action)
            # Execute action in environment and get reward
            reward, b = sac.reward_function(action)
            count += 1
            if count >= 5:
                if b == True:
                    print("###HERE###")
                    t = t - 1
                    break

        t += 1

        action_tuple = tuple(action[0]) if isinstance(action, np.ndarray) else tuple(action)
        rewards[action_tuple] = reward
        active_arms.append(action_tuple)
        values.append(reward)
        agent.update(state_tensor, action, reward)

    best_index = np.argmax([rewards[arm] for arm in active_arms])
    best_prompt = active_arms[best_index]
    # best_prompt =np.array(best_prompt)
    best_reward = rewards[best_prompt]
    best_values.append(best_reward)

    # Evaluate on test data
    print('Evaluating on test data...')
    prompts_set = model_forward_api.return_prompts_set()
    # print("The final instruction set is:")
    # print(prompts_set)
    logging.info(f"all prompts are {prompts_set}")
    prompts_list = model_forward_api.return_prompts_list()
    prompts = model_forward_api.return_best_prompt()
    logging.info(f'Get the best instruction {prompts}')
    test_res = ape.evaluate_prompts(prompts=prompts,
                                    eval_template=eval_template,
                                    eval_data=test_data,
                                    few_shot_data=prompt_gen_data,
                                    demos_template=demos_template,
                                    conf=test_conf,
                                    base_conf=base_conf)
    test_res = test_res[0]
    test_score = test_res.sorted()[1][0]
    logging.info(f"the best prompt is {prompts}")
    logging.info(f"the test score is {test_score}")
    logging.info(f"the prompt achieved reward equals to {best_values}")
    return test_score, prompts, prompts_list, best_values
    # print(f'Test score on ChatGPT: {test_score}')


def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )

    parser.add_argument(
        "--rep",
        type=bool,
        default=False,
        help="Use repetition of best prompts.",
    )

    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default="/datasets/Samsung/samsum",
        help="Your vicuna directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Set the parameter nu."
    )
    parser.add_argument(
        "--lamdba",
        type=float,
        default=0.1,
        help="Set the lamdba parameter."
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=40,
        help="Set the number of initialization points."
    )
    parser.add_argument(
        "--n_domain",
        type=int,
        default=10000,
        help="Set the number of domain."
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=165,
        help="Set the number of total queries."
    )
    parser.add_argument(
        "--local_training_iter",
        type=int,
        default=30,
        help="Set the number of total queries."
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default='uniform',
        help="Set the projection method."
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=100,
        help="Set the number of intrinsic dim."
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="Set the number of domains to be evaluated at each ucb iteration."
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Set the name of the experiments."
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="gpt-3.5-turbo",
        help="Which version of gpt to use."
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=1,
        help="Which scale to use."
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        help="Which pooling method to use."
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(args.seed))
    print("###############################")
    test_score, prompts, prompts_list, best_values = run(
        task=args.task,
        n_prompt_tokens=args.n_prompt_tokens,
        HF_cache_dir=args.HF_cache_dir,
        total_iter=args.total_iter,
        random_proj=args.random_proj,
        intrinsic_dim=args.intrinsic_dim,
        gpt=args.gpt,
    )

    args_dict = vars(args)
    args_dict['test_score'] = test_score
    args_dict['best_prompt'] = prompts
    args_dict['prompts_list'] = prompts_list
    args_dict['best_values'] = best_values
    print("args_dict['test_score']", args_dict['test_score'])
    print("args_dict['best_prompt']", args_dict['best_prompt'])
    print("args_dict['best_values']", args_dict['best_values'])

    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')

    # best_r = 0
    best_values = []
