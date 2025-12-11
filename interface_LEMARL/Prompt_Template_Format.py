PROMPT_TEMPLATE = """
You are an expert energy system optimization engineer. Your task is to generate Python functions for Multi-Agent Reinforcement Learning (MARL) in Integrated Energy System (IES) scheduling.

SYSTEM DESCRIPTION:
{system_description}

CONSTRAINTS:
{constraints}

OBJECTIVE WEIGHTS:
{objective_weights}

STATISTICAL DATA:
{statistics}

AVAILABLE APIs:
{api_functions}

TASK:
Generate two Python functions:

1. Prior Policy Function: generate_prior_policy(observation) -> action
   - Input: observation dict containing {observation_keys}
   - Output: action dict for each agent type
   - Requirements:
     * Must satisfy basic power/thermal balance
     * Must respect output limits and ramping constraints
     * Should follow a reasonable dispatch logic (e.g., merit order)

2. Reward Function**: compute_reward(state, action) -> float
   - Input: state dict and action dict
   - Output: scalar reward in range [0, 1]
   - Requirements:
     * Return 1.0 if all sub-objectives are satisfied
     * Return 0.0 otherwise
     * Sub-objectives based on objective_weights: {sub_objectives}

CHAIN-OF-THOUGHT INSTRUCTIONS:
Please follow these steps:
1. Identify basic operational skills needed (e.g., load balancing, ramping control)
2. Categorize constraints into basic (simple logic) and complex (multi-objective coupling)
3. Determine coupling relationships between units
4. List key sub-objectives that must be included in the reward function
5. Generate the two functions with detailed comments

OUTPUT FORMAT:
Provide your response in the following JSON format:
{{
  "reasoning": {{
    "basic_skills": ["skill1", "skill2", ...],
    "constraint_categories": {{"basic": [...], "complex": [...]}},
    "coupling_analysis": "text description",
    "key_subobjectives": ["obj1", "obj2", ...]
  }},
  "prior_policy_code": "complete Python function as string",
  "reward_function_code": "complete Python function as string",
  "explanation": "natural language explanation of the logic"
}}

IMPORTANT NOTES:
- Use only the provided APIs for state information
- Ensure all array indexing is safe (check bounds)
- Handle edge cases (e.g., division by zero)
- Add error handling with try-except blocks
- All numeric values should be float type
"""

def format_prompt(input_json):
    return PROMPT_TEMPLATE.format(
        system_description=json.dumps(input_json['system_description'], indent=2),
        constraints=json.dumps(input_json['constraints'], indent=2),
        objective_weights=json.dumps(input_json['objective_weights'], indent=2),
        statistics=json.dumps(input_json['statistics'], indent=2),
        api_functions=get_api_documentation(),
        observation_keys=extract_observation_keys(input_json),
        sub_objectives=extract_sub_objectives(input_json['objective_weights'])
    )