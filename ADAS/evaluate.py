import time
import subprocess
import os
import json
import openai
import anthropic
from typing import Tuple, Optional, List

from pathlib import Path
from nyuctf.challenge import CTFChallenge

from llm_ctf.ctflogging import status
from llm_ctf.backends import Backend
from llm_ctf.prompts.prompts import PromptManager
from llm_ctf.tools import ToolCall, ToolResult, Tool, TOOLSETS
from llm_ctf.environment import CTFEnvironment

now = lambda: time.time()

class CTFConversation:
    def __init__(self, environment: CTFEnvironment, challenge: CTFChallenge, prompt_manager: PromptManager, backend: Backend, logfile: Path, max_rounds:int=30, max_cost:float=1.0, args=None):
        self.challenge = challenge
        self.environment = environment
        self.prompt_manager = prompt_manager
        self.backend = backend
        self.logfile = logfile

        self.available_functions : dict[str,Tool] = {}

        self.max_rounds = max_rounds
        self.max_cost = max_cost
        # self.config = config
        self.args = args

        self.rounds = 0
        self.cost = 0
        self.finish_reason = "unknown"
        self.model_time = 0
        self.tool_time = 0

    def __enter__(self):
        self.backend.setup()
        self.challenge.start_challenge_container()
        self.environment.setup()

        self.start_time = now()
        return self

    def run(self):
        next_msg = self.prompt_manager.initial_message(self.challenge)
        while not self.environment.solved and not self.environment.giveup \
                and self.rounds <= self.max_rounds and self.cost <= self.max_cost:
            try:
                tools_run = self.run_conversation_step(next_msg)
                if tools_run == 0:
                    next_msg = "Please proceed to the next step using your best judgment."
                else:
                    next_msg = None
            except KeyboardInterrupt:
                status.print("[red bold]Interrupted by user[/red bold]", markup=True)
                self.finish_reason = "user_cancel"
                break
            # TODO Normalize the ratelimiterrors
            except (openai.RateLimitError, anthropic.RateLimitError):
                status.print("[red bold]Rate limit reached![/red bold]", markup=True)
                self.finish_reason = "rate_limit"
                break
            except openai.BadRequestError as e:
                msg = str(e)
                if "'code': 'context_length_exceeded'" in msg or "'code': 'string_above_max_length'" in msg:
                    status.print("[red bold]Context length exceeded![/red bold]", markup=True)
                    self.finish_reason = "context_length"
                    break
                else:
                    # Some other error, re-raise
                    raise

        # Look for a finish reason
        if self.environment.solved:
            status.print("[red bold]Challenge solved by our robot overlords![/red bold]", markup=True)
            self.finish_reason = "solved"
        elif self.environment.giveup:
            status.print("[red bold]The LLM decided to give up! NGMI.[/red bold]", markup=True)
            self.finish_reason = "give_up"
        elif self.cost > self.max_cost:
            status.print(f"[red bold]Challenge is unsolved after {self.max_cost} dollars of cost; exiting[/red bold]", markup=True)
            self.finish_reason = "max_cost"
        elif self.rounds > self.max_rounds:
            status.print(f"[red bold]Challenge is unsolved after {self.max_rounds} rounds; exiting[/red bold]", markup=True)
            self.finish_reason = "max_rounds"

    def run_tools(self, tool_calls: List[ToolCall]) -> Tuple[Optional[str],bool]:
        tool_results = []
        for tool_call in tool_calls:
            # Tool lookup
            tool = self.environment.available_tools.get(tool_call.name)
            if not tool:
                status.error_message(f"Unknown tool {tool_call.name}")
                tool_results.append(tool_call.error(f"Unknown tool {tool_call.name}"))
                continue

            # Parse arguments
            parsed, tool_call = self.backend.parse_tool_arguments(tool, tool_call)
            if not parsed:
                tool_results.append(tool_call)
                continue

            try:
                tool_res = tool.run(tool_call)
            except TypeError as e:
                status.debug_message(f"Error encoding results from {tool.name}: {e}")
                tool_res = tool_call.error(f"{type(e).__name__} running {tool.name}: {e}")
            except Exception as e:
                status.debug_message(f"Error running {tool.name}: {e}")
                tool_res = tool_call.error(f"{type(e).__name__} running {tool.name}: {e}")
            tool_results.append(tool_res)
        return tool_results

    def run_conversation_step(self, message: Optional[str]=None):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = now()
        self.environment.teardown(exc_type, exc_value, traceback)
        self.challenge.stop_challenge_container()
        if self.backend.NAME != 'vllm':
            messages = self.backend.get_timestamped_messages()
        else:
            orig_messages = self.backend.outgoing_messages
            messages = []
            for m in orig_messages:
                if isinstance(m, dict):
                    pass
                else:
                    m = {"role": "assistant", "content": m.content}
                messages.append(m)
            

        self.logfile.write_text(json.dumps(
            {
                "args": vars(self.args),
                "messages": messages,
                "challenge": self.challenge.challenge_info,
                "solved": self.environment.solved,
                "rounds": self.rounds,
                "cost": self.cost,
                "debug_log": status.debug_log,
                # "challenge_server_output": self.chal.challenge_server_output,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "runtime": {
                    "total": self.end_time - self.start_time,
                    "tools": self.tool_time,
                    "model": self.model_time
                },
                # "exception_info": exception_info,
                "finish_reason": self.finish_reason,
            },
            indent=4
        ))
        status.print(f"Conversation saved to {self.logfile}")


def evaluate_forward_fn(forward_str):
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(CTFConversation, "run_conversation_step", func)
