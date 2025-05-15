import json
import re

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def run_conversation_step(self, message: List) -> List:
    # Your code here
    return answer
"""
}

Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. After each answer, testing on the examples to provide feedback, and the LLM uses insights from previous attempts and feedback to refine its answer. We send the message to the backend and get the feedback from the environment if there's a tool call in the message. One should consider trying to use this feedback in future agent design.",
    "name": "Initial-Simple-ToolRun-Feedback",
    "code": """def run_conversation_step(self, message: List) -> List:
        response, tool_calls = self.send_message(message)
        message.append({"role": "assistant", "content": response})
        self.rounds += 1

        # Run tool calls
        if tool_calls:
            tool_results = self.run_tools(tool_calls)
            next_msg = tool_results
        else:
            next_msg = "Please proceed to the next step using your best judgment."
        
        message.append({"role": "user", "content": next_msg})
            
        return message"""
}


system_prompt = """
You are an expert *agent designer*.  
Respond **with one and only one well‑formed JSON object** that has **exactly three** top‑level keys:

1. **"thought"** – your reasoning in Markdown:  
   • why this new agent should work better than previous ones, and  
   • a step‑by‑step plan for implementing it.  

2. **"name"** – a short, memorable agent name.  

3. **"code"** – a complete Python definition of  

       def run_conversation_step(self, message: List) -> List:
    
   that obeys the existing project API.

Output *nothing* outside that JSON. **Think like you’re inventing a new agent architecture, not tweaking old code.**
"""


base = """## Goal
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an improved implementation of `CTFConversation.run_conversation_step` that maximises the fraction of CTF (Capture-the-flag) challenges solved.

## Environment recap
* Each challenge provides files, a description, and an interactive tool API.  

* Talk to the LLM with  

      response, tool_calls = self.send_message(message)

  – `response` is the raw assistant reply, `tool_calls` is any tool‑use the LLM proposed.  
  - Use the following code to append the response.

      message.append({"role": "assistant", "content": response})
  – You may call **send_message** several times in the same round to explore different ideas and **Pick exactly one** of those candidate replies and append it to the history. This is not mandatory, you are encouraged to explore different workflows. 

* If the chosen reply contains tool calls, run them with  

      tool_results = self.run_tools(tool_calls)

  - Use the following code to append the tool_results.

      message.append({"role": "user", "content": tool_results})
      
  - You may execute **run_tools** multiple times (e.g., on alternative `tool_calls` lists) and compare the outputs, append the most useful tool_results together with its corresponding assistant response to the message list.  This is not mandatory, you are encouraged to explore different workflows.

* In every round you **must** append one assistant message (and, if tools were used, one tool‑results message) before looping again.

* Only use this two functions in your implementation. Do not call any other functions or methods that are not defined in the code. Never use functions related to `self.prompt_manager`, it is just a rendering tool and can only generate the initial challenge message.

* We set the max_rounds=20, which means len(message) <= 20.

* The context window of the agent is 120000 tokens.


## conversation.py API (read‑only)

```python
import time
import json
import openai
from typing import List, Tuple, Optional, Union, Any
from pathlib import Path
from nyuctf.challenge import CTFChallenge
from llm_ctf.formatters import Formatter
from .ctflogging import status
from .tools import Tool, ToolCall, ToolResult


class CTFConversation:
    def __init__(self, environment, challenge: CTFChallenge, prompt_manager, logfile: Path, 
                 max_rounds:int=20, args=None, tools=None):
        self.challenge = challenge
        self.environment = environment
        self.prompt_manager = prompt_manager
        self.logfile = logfile
        self.args = args

        # VLLM specific configuration
        self.model = args.model
        self.temperature = args.temperature
        self.formatter = Formatter.from_name("xml")(tools, prompt_manager)
        
        # Runtime metrics
        self.max_rounds = max_rounds
        self.rounds = 0
        self.finish_reason = "unknown"
        
        # Prepare system messages
        self.system_message_content = prompt_manager.system_message(challenge)
        tool_use_prompt = self.formatter.tool_use_prompt()
        self.system_message_content += '\n\n' + tool_use_prompt
        self.initial_challenge_message = prompt_manager.initial_message(challenge)
        
        # Setup OpenAI client for calling vLLM
        self.client = openai.OpenAI(api_key="token-abc123", base_url=f"http://localhost:6790/v1")

    def run(self):
        message = []
        message.append({"role": "system", "content": self.system_message_content})
        message.append({"role": "user", "content": self.initial_challenge_message})
        while not self.environment.solved and self.rounds <= self.max_rounds:
            try:
                message = self.run_conversation_step(message)
                self.outgoing_messages = message
            except KeyboardInterrupt:
                status.print("[red bold]Interrupted by user[/red bold]", markup=True)
                self.finish_reason = "user_cancel"
                break
            except openai.BadRequestError as e:
                msg = str(e)
                if "maximum context length" in msg:
                    status.print("[red bold]Context length exceeded![/red bold]", markup=True)
                    self.finish_reason = "context_length"
                    break
                else:
                    raise

        # Check finish reason
        if self.environment.solved:
            status.print("[red bold]Challenge solved by our robot overlords![/red bold]", markup=True)
            self.finish_reason = "solved"
        elif self.rounds > self.max_rounds:
            status.print(f"[red bold]Challenge is unsolved after {self.max_rounds} rounds; exiting[/red bold]", markup=True)
            self.finish_reason = "max_rounds"
            

    def send_message(self, message: List) -> Tuple[str, List[ToolCall]]:
        
        # Get the formatter delimiters
        start_seqs = ['<function_calls>']
        stop_seqs = ['</function_calls>']
        
        # Call the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=self.temperature,
            max_tokens=1024,
            stop=stop_seqs,
            extra_body={'repetition_penalty': 1.0},
        )
        
        # Process the response
        message = response.choices[0].message
        content = message.content
        
        # Check if the model wants to run tools
        has_tool_calls = False
        if response.choices[0].finish_reason == "stop" and any(s in content for s in start_seqs):
            content += "\n" + stop_seqs[0] + "\n"
            has_tool_calls = True
        
        # Process tool calls if any
        tool_calls = []
        if has_tool_calls:
            try:
                tool_calls = self.formatter.extract_tool_calls(content)
            except Exception as e:
                pass
        
        return content, tool_calls

    def parse_tool_arguments(self, tool: Tool, tool_call: ToolCall) -> Tuple[bool, Union[ToolCall, ToolResult]]:
        # Already parsed arguments
        if tool_call.parsed_arguments:
            return True, tool_call
            
        try:
            parsed_tc = self.formatter.extract_params(tool, tool_call)
            # Update the tool call with parsed arguments
            tool_call.parsed_arguments = parsed_tc.parsed_arguments
            return True, tool_call
        except ValueError as e:
            msg = f"{type(e).__name__} extracting parameters for {tool_call.name}: {e}"
            status.debug_message(msg)
            return False, tool_call.error(msg)

    def run_tools(self, tool_calls: List[ToolCall]) -> str:
        tool_results = []
        for tool_call in tool_calls:
            # Tool lookup
            tool = self.environment.available_tools.get(tool_call.name)
            if not tool:
                status.error_message(f"Unknown tool {tool_call.name}")
                tool_results.append(tool_call.error(f"Unknown tool {tool_call.name}"))
                continue

            # Parse arguments
            parsed, result = self.parse_tool_arguments(tool, tool_call)
            if not parsed:
                tool_results.append(result)
                continue

            # Run the tool
            try:
                tool_res = tool.run(tool_call)
            except TypeError as e:
                status.debug_message(f"Error encoding results from {tool.name}: {e}")
                tool_res = tool_call.error(f"{type(e).__name__} running {tool.name}: {e}")
            except Exception as e:
                status.debug_message(f"Error running {tool.name}: {e}")
                tool_res = tool_call.error(f"{type(e).__name__} running {tool.name}: {e}")
            
            tool_results.append(tool_res)
            
        # Format the results and add to conversation
        tool_results_msg = self.formatter.tool_result_prompt(tool_results)
        
        return tool_results_msg



    \"""
    Fill in your code here.
    \"""
    def run_conversation_step(self, message: List) -> List:
        \"""
        Placeholder method for processing task information.

        Returns:
        - message (List): The input message (a list of conversation history) for the next round of conversation. 
        \"""     
        pass
```

Treat everything in this block as read‑only except for rewriting the body of run_conversation_step.

## Constraints
* **Touch only** `run_conversation_step`; do **not** edit other classes.  
* Use only the functions and variables already defined in the snippet above.  
* Do **not** add flag‑checking logic—the framework handles success detection.  
* Follow the “## WRONG Implementation examples” section to avoid common errors.

## Evaluation
Every JSON you output counts as a new *generation*; the metric is accuracy on held‑out CTF tasks.


## Discovered architecture archive
Here is the archive of the discovered architectures.

[ARCHIVE]

## WRONG Implementation examples:
Here are some mistakes you may make:
1. This is WRONG: ```python
self.send_message(message)
```
Do not call `self.send_message(message)` alone. Instead, always use this function using assignment statement: `response, tool_calls=self.send_message(message)` and utilze the return values.

2. This is WRONG: ```python
message = self.prompt_manager.generate_hypothesis(self.challenge) # or other non-defined function
```
You cannot use any function that is not defined in the code. Never use functions related to `self.prompt_manager`, it is just a rendering tool and can only generate the initial challenge message. You can only use the functions that are defined in the code. You should not assume any function is defined if not shown in the code.

3. This is WRONG: ```python
def run_conversation_step(self, message: List) -> List:
    # Your code
def <other functions>(self, message):
    # Your code
```
You should not define any other functions except `run_conversation_step`. Only return the code in the `run_conversation_step` function.

4. This is WRONG: ```python
if 'sub-tasks:' in response.lower():
    # Your code
```
There are no subtasks in the CTF task, and the agent will not generate subtasks. Avoid using keyword detection in your implementation.

5. This is WRONG: ```python
if last_assistant_msg.startswith('<function_calls>'):
    # Extract tool calls and run them
    tool_calls = self.formatter.extract_tool_calls(last_assistant_msg)
```
Calling `self.send_message(message)` will automatically extract the function calls from the response. You don't need to extract the function calls again in `run_conversation_step`.

DON'T make those mistakes.

## Your task
You are deeply familiar with prompting techniques and the agent works from the literature. Your goal is to maximize the specified performance metrics by proposing interestingly new agents.
Observe the discovered agents carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative when thinking about the next interesting agent to try.

THINK OUTSIDE THE BOX.

"""

Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""

def fitness_value(d):
    m = re.search(r"(\d+\.?\d*)\s*%", d["fitness"])
    return float(m.group(1)) if m else float("0.00")


def get_prompt(current_archive, adaptive=False):
    sorted_archive = sorted(current_archive[1:], key=fitness_value, reverse=True)[:10] # we always keep the initial solution
    sorted_archive = sorted(sorted_archive, key=lambda d: float(d["generation"]))
    sorted_archive = current_archive[:1] + sorted_archive
    archive_str = ",\n".join([json.dumps(sol) for sol in sorted_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    return system_prompt, prompt


def get_init_archive():
    return [Reflexion]


def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2
