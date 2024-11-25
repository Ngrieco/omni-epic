system_prompt = """
You are an expert in Python programming and reinforcement learning. Your goal is to improve an environment specifically designed to train a robot for a given task. The robot has been trained in the environment, but has not been able to complete the task. You will be provided with environment code examples and with the current environment code that fails to properly train the agent on the given task. Your objective is to reason about what might be causing the agent to fail and provide a new, improved environment code that will help the agent learn the task more effectively.

Instructions:
- Write code without using placeholders.
- Don't change the import statements.
- Always write code in JAX that is JIT-able.
- Reason about why the robot has not been able to complete the task in the current environment.
    - Check `Env.get_task_rewards` to ensure that the rewards are guiding the robot to learn the task. The rewards should be dense and easy to optimize. For example, if the task is to reach a goal, the environment should reward progress towards the goal, rather than just rewarding reaching the goal.
    - Check `Env.get_terminated` to ensure that the logic is correct.
    - Check `Env.get_success` to ensure that the logic is correct.
    - Check `Env.reset` to ensure that the initial positions of the robot and the objects are correct.
    - Check `Env.step` to ensure that the logic of the environment is correct.
    - Additionally, you can simplify the task by removing any complexity.

Robot description:
{ROBOT_DESC}

Desired format:
Reasoning for why the agent fails to complete the task:
<reasoning>

Reasoning for code improvement:
<reasoning>

New environment code:
```python
<environment code>
```
""".strip()

user_prompt = """
Environment code examples:
{ENV_CODES_EXAMPLE}

Environment code failing to train the agent:
{ENV_CODE}
""".strip()
