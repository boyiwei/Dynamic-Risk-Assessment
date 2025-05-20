from datetime import datetime
import socket
import subprocess
import time
import json
import os
import argparse
import shutil
import yaml
from pathlib import Path

from nyuctf.dataset import CTFDataset
from nyuctf.challenge import CTFChallenge

from llm_ctf.ctflogging import status
from llm_ctf.backends import Backend, OpenAIBackend, AnthropicBackend, VLLMBackend
from llm_ctf.formatters import Formatter
from llm_ctf.prompts.prompts import PromptManager
from llm_ctf.environment import CTFEnvironment

from nyuctf.dataset import CTFDataset
from nyuctf.challenge import CTFChallenge

import os
import json
import openai
from typing import Tuple, Optional, List, Union

from pathlib import Path
from nyuctf.challenge import CTFChallenge
from llm_ctf.ctflogging import status
from llm_ctf.backends import Backend
from llm_ctf.prompts.prompts import PromptManager
from llm_ctf.tools import ToolCall, ToolResult, Tool, TOOLSETS
from llm_ctf.environment import CTFEnvironment

now = lambda: time.time()

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


    def __enter__(self):
        self.challenge.start_challenge_container()
        self.environment.setup()
        self.start_time = now()
        return self

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

    def run_conversation_step(self, message: List) -> List:
        # Prompt the model to produce a response and tool_calls
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
            
        return message

    def send_message(self, message: str) -> Tuple[str, List[ToolCall]]:
        """Send a message to the model and get response with optional tool calls"""
        
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
        """Parse arguments for a tool call"""
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
        """Run all tools in the tool calls and return formatted results"""
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

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = now()
        self.environment.teardown(exc_type, exc_value, traceback)
        self.challenge.stop_challenge_container()
        
        
        # Write log file
        self.logfile.write_text(json.dumps(
            {
                "args": vars(self.args),
                "messages": self.outgoing_messages,
                "challenge": self.challenge.challenge_info,
                "solved": self.environment.solved,
                "rounds": self.rounds,
                "debug_log": status.debug_log,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "runtime": {
                    "total": self.end_time - self.start_time,
                },
                "finish_reason": self.finish_reason,
            },
            indent=4
        ))
        status.print(f"Conversation saved to {self.logfile}")



def main():
    parser = argparse.ArgumentParser(
        description="Use an LLM to solve a CTF challenge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    model_list = []
    for b in Backend.registry.values():
        model_list += b.get_models()
    model_list = list(set(model_list))

    script_dir = Path(__file__).parent.resolve()

    parser.add_argument("--challenge", required=True, help="Name of the challenge")
    parser.add_argument("--dataset", help="Dataset JSON path. Only provide if not using the NYUCTF dataset at default path")
    parser.add_argument("-s", "--split", default="development", choices=["test", "development"], help="Dataset split to select. Only used when --dataset not provided.")
    parser.add_argument("-c", "--config", help="Config file to run the experiment")

    parser.add_argument("-q", "--quiet", action="store_true", help="don't print messages to the console")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug messages")
    parser.add_argument("-M", "--model", help="the model to use (default is backend-specific)", choices=model_list)
    parser.add_argument("-C", "--container-image", default="ctfenv", help="the Docker image to use for the CTF environment")
    parser.add_argument("--container-name", default="ctf_env", help="the Docker container name to use for the CTF environment")
    parser.add_argument("-N", "--network", default="ctfnet", help="the Docker network to use for the CTF environment")
    parser.add_argument("--api-key", default=None, help="API key to use when calling the model")
    parser.add_argument("--api-endpoint", default=None, help="API endpoint URL to use when calling the model")
    parser.add_argument("--backend", default="openai", choices=Backend.registry.keys(), help="model backend to use")
    parser.add_argument("--formatter", default="xml", choices=Formatter.registry.keys(), help="prompt formatter to use")
    parser.add_argument("--prompt-set", default="default", help="set of prompts to use")
    # TODO add back hints functionality
    parser.add_argument("--hints", default=[], nargs="+", help="list of hints to provide")
    parser.add_argument("--disable-markdown", default=False, action="store_true", help="don't render Markdown formatting in messages")
    parser.add_argument("-m", "--max-rounds", type=int, default=10, help="maximum number of rounds to run")
    parser.add_argument("--max-cost", type=float, default=10, help="maximum cost of the conversation to run")
    parser.add_argument("--temperature", type=float, default=0.6, help="temperature for sampling")

    # Log directory options
    parser.add_argument("--skip-exist", action="store_true", help="Skip existing logs and experiments")
    parser.add_argument("-L", "--logdir", default=str(script_dir / "logs"), help="log directory to write the log")
    parser.add_argument("-n", "--name", help="Experiment name (creates subdir in logdir)")
    parser.add_argument("-i", "--index", help="Round index of the experiment (creates subdir in logdir)")
    parser.add_argument("--adas-iter-round", type=int, default=1, help="The round index of the ADAS iteration to use")

    args = parser.parse_args()
    config = None
    if args.config:
        try:
            with open(args.config, "r") as c:
                config = yaml.safe_load(c)
        except FileNotFoundError:
            pass

    if config:
        config_parameter = config.get("parameter", {})
        config_experiment = config.get("experiment", {})
        config_demostration = config.get("demostration", {})

        if not args.max_rounds:
            args.max_rounds = config_parameter.get("max_rounds", args.max_rounds)
        args.backend = config_parameter.get("backend", args.backend)
        if not args.model:
            args.model = config_parameter.get("model", None)
        # args.model = config_parameter.get("model", args.model)
        args.max_cost = config_parameter.get("max_cost", args.max_cost)
        if not args.name:
            args.name = config_experiment.get("name", None)
        # args.name = config_experiment.get("name", args.name)
        args.debug = config_experiment.get("debug", args.debug)
        args.skip_exist = config_experiment.get("skip_exist", args.skip_exist)
        args.hints = config_demostration.get("hints", [])
    
    print(args)

    status.set(quiet=args.quiet, debug=args.debug, disable_markdown=args.disable_markdown)

    if args.dataset is not None:
        dataset = CTFDataset(dataset_json=args.dataset)
    else:
        dataset = CTFDataset(split=args.split)
    challenge = CTFChallenge(dataset.get(args.challenge), dataset.basedir)

    logdir = Path(args.logdir).expanduser().resolve()
    logsubdir = []
    if args.name:
        logsubdir.append(args.name)
    if args.index:
        logsubdir.append(f"round{args.index}")
    if len(logsubdir) > 0:
        logdir = logdir / ("_".join(logsubdir))
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"{challenge.canonical_name}.json"
    
    if logfile.exists() and args.skip_exist:
        status.print(f"[red bold]Challenge log {logfile} exists; skipping[/red bold]", markup=True)
        exit()
        
    print(f"Using model: {args.model}")
    print(f"Output path: {logfile}")
    print(f"Max rounds: {args.max_rounds}")
    
    environment = CTFEnvironment(challenge, args.container_image, args.network, args.container_name)
    prompt_manager = PromptManager(prompt_set=args.prompt_set, config=config)

    if args.backend == "openai":
        backend = OpenAIBackend(
                        prompt_manager.system_message(challenge),
                        prompt_manager.hints_message(),
                        environment.available_tools,
                        model=args.model,
                        api_key=args.api_key,
                        args=args
                    )
    elif args.backend ==  "anthropic":
        backend = AnthropicBackend(
                        prompt_manager.system_message(challenge),
                        prompt_manager.hints_message(),
                        environment.available_tools,
                        prompt_manager,
                        model=args.model,
                        api_key=args.api_key,
                        args=args
                    )
    elif args.backend == "vllm":
        backend = VLLMBackend(
                        prompt_manager.system_message(challenge),
                        prompt_manager.hints_message(),
                        environment.available_tools,
                        prompt_manager,
                        model=args.model,
                        api_key=args.api_key,
                        api_endpoint=args.api_endpoint,
                        formatter=args.formatter,
                        args=args
                    )
    
    adas_file_path = "iter_workflow_refinement/ctf_results_run_archive.json"
    with open(adas_file_path, "r") as f:
        adas_file = json.load(f)
    forward_str = adas_file[args.adas_iter_round]['code'] # get the latest agent workflow
    print(f"ADAS Iter Round: {args.adas_iter_round}, Using forward function: {forward_str}")
    namespace = {}
    exec(forward_str, globals(), namespace)
    names = list(namespace.keys())
    if len(names) != 1:
        raise AssertionError(f"{len(names)} things in namespace. Please only provide 1")
    func = namespace[names[0]]
    if not callable(func):
        raise AssertionError(f"{func} is not callable")
    setattr(CTFConversation, "run_conversation_step", func)
    print("successfully set the forward function")
    
    tools = environment.available_tools
    with CTFConversation(environment, challenge, prompt_manager, logfile, max_rounds=args.max_rounds, args=args, tools=tools) as convo:
        convo.run()

if __name__ == "__main__":
    main()
