from .utils import *
from argparse import Namespace
from pathlib import Path

from llm_ctf.formatters.vbpy import VBPYFormatter
from ..formatters.formatter import Formatter
from .backend import (AssistantMessage, Backend, ErrorToolCalls, FakeToolCalls,
                      SystemMessage, UnparsedToolCalls, UserMessage, HintMessage)
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from ..tools import Tool, ToolCall, ToolResult
from rich.pretty import Pretty

class VLLMBackend(Backend):
    NAME = 'vllm'
    MODELS = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "deepseek-coder-33b-instruct",
        "llama3:70b-instruct-fp16",
        "wizardlm2:8x22b-q8_0",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "Qwen2.5-Coder-32B-Instruct",
        "Qwen2.5-Coder-32B-Instruct-ft",
    ]
    QUIRKS = {
        "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelQuirks(
            supports_system_messages=False,
            needs_tool_use_demonstrations=True,
            clean_content=fix_xml_tag_names,
            clean_tool_use=fix_xml_tag_names,
            augment_stop_sequences=fix_xml_seqs,
            augment_start_sequences=fix_xml_seqs,
        ),
        "llama3:70b-instruct-fp16": ModelQuirks(
            supports_system_messages=True,
            augment_stop_sequences=lambda seqs: seqs + ["<|eot_id|>"],
        ),
        "meta-llama/Meta-Llama-3-70B-Instruct": ModelQuirks(
            supports_system_messages=True,
            augment_stop_sequences=lambda seqs: seqs + ["<|eot_id|>"],
        ),
    }

    def __init__(self, system_message : str, hint_message: str, tools: dict[str,Tool], prompt_manager, model=None, api_key=None, api_endpoint=None, formatter="xml", args: Namespace = None):
        self.formatter : Formatter = Formatter.from_name("xml")(tools, prompt_manager)
        self.tools = tools
        self.args = args

        if model:
            if model not in self.MODELS:
                raise ValueError(f"Invalid model {args.model} for backend. Must be one of {self.MODELS}")
            self.model = model
        else:
            self.model = self.MODELS[0]
        self.api_key = api_key
        self.api_endpoint = api_endpoint

        self.client_setup()
        self.quirks = self.QUIRKS.get(self.model, NO_QUIRKS)
        self.system_message_content = system_message
        self.hint_message_content = hint_message
        self.outgoing_messages = []
        self.last_tool_calls = []

    def client_setup(self):
        # if self.api_endpoint:
        #     base_url = self.api_endpoint
        # elif "MODEL_URL" in KEYS:
        #     base_url = KEYS["MODEL_URL"].strip()
        # else:
        #     raise ValueError(f"No VLLM Endpoint provided")
        self.client = OpenAI(api_key="token-abc123", base_url=f"http://localhost:6790/v1")

    @classmethod
    def get_models(cls):
        return cls.MODELS

    def setup(self):
        # Add initial messages
        tool_use_prompt = self.formatter.tool_use_prompt()
        self.messages.append(SystemMessage(self.system_message_content, tool_use_prompt))
        if self.args.hints and self.hint_message_content:
            self.messages.append(HintMessage(self.hint_message_content))
        self.system_message_content += '\n\n' + tool_use_prompt
        status.system_message(self.system_message_content)

        if self.quirks.supports_system_messages:
            system_messages = [
                self.system_message(self.system_message_content),
            ]
        else:
            system_messages = [
                self.user_message(self.system_message_content),
                self.assistant_message("Understood."),
            ]
        if self.args.hints and self.hint_message:
            system_messages += [self.hint_message(self.hint_message_content), self.assistant_message("Understood.")]
            status.hint_message(self.hint_message_content)

        # import pdb; pdb.set_trace()
        for sm in system_messages:
            self.append(sm)

        # TODO this won't work because run_tools has moved to Conversation
        # if self.quirks.needs_tool_use_demonstrations:
        #     num_messages = len(self.outgoing_messages)
        #     self.make_demonstration_messages()
        #     # Print them out
        #     if self.args.debug:
        #         for message in self.outgoing_messages[num_messages:]:
        #             if message['role'] == 'user':
        #                 status.user_message(message['content'])
        #             elif message['role'] == 'assistant':
        #                 content = message['content']
        #                 if self.formatter.get_delimiters()[0][0] in message['content']:
        #                     content = "🤔 ...thinking... 🤔\n\n" + content
        #                 status.assistant_message(content)

    # TODO won't work without run_tools, make this common across backends
    # def demo_tool_call(self, tool_name: str, args: dict[str, Any], out:List) -> Literal[""]:
    #     tool = self.tools[tool_name]
    #     tool_call = tool.make_call(**args)
    #     out.append(tool_call)
    #     return ""
    # def tool_demo(self, template):
    #     tool_calls: List[ToolCall] = []
    #     def render_tool_calls():
    #         return self.prompt_manager.tool_calls(self.formatter, tool_calls)
    #     tool_calls_content = self.prompt_manager.render(
    #         template,
    #         make_tool_call=self.demo_tool_call,
    #         dest=tool_calls,
    #         render_tool_calls=render_tool_calls,
    #     )
    #     self.messages.append(FakeToolCalls(tool_calls))
    #     self.append(self.assistant_message(tool_calls_content))
    #     tool_results = self.run_tools_internal(tool_calls, demo=True)
    #     self.append(self.tool_results_message(tool_results))
        # NB: the tool results are not added to self.messages because they are
        # added inside of _run_tools_internal.
    # def make_demonstration_messages(self):
    #     role_states = {
    #         'user': {'assistant', 'tool'},
    #         'tool': {'assistant', 'tool'},
    #         'assistant': {'user'},
    #     }
    #     demo_templates = self.prompt_manager.env.list_templates(
    #         filter_func=lambda s: f'{self.args.prompt_set}/demo_messages' in s
    #     )
    #     expected_next_roles = {"user"}
    #     for msg_num, template in enumerate(sorted(demo_templates)):
    #         template_name = Path(template).name
    #         match re.match(r'(\d\d)_(user|assistant|tool)', template_name):
    #             case None:
    #                 status.debug_message(f"Warning: demo message template {template} doesn't "
    #                                      f"match expected format; skipping")
    #                 continue
    #             case m:
    #                 num, role = m.groups()
    #                 template_stem = f"demo_messages/{m.group(0)}"

    #         # Do some validation
    #         if role not in role_states:
    #             status.debug_message(f"Warning: demo message template {template} has "
    #                                     f"unexpected role {role}; skipping")
    #             continue
    #         if int(num) != msg_num:
    #             status.debug_message(f"Warning: demo message template {template} has "
    #                                     f"unexpected number {num}")
    #         if role not in expected_next_roles:
    #             status.debug_message(f"Warning: demo message template {template} has "
    #                                     f"unexpected role {role}; expected one of {expected_next_roles}")
    #         expected_next_roles = role_states[role]

    #         # Process the demo message
    #         match role:
    #             case "user":
    #                 content = self.prompt_manager.render(template_stem)
    #                 self.messages.append(UserMessage(content))
    #                 self.append(self.user_message(content))
    #             case "assistant":
    #                 content = self.prompt_manager.render(template_stem)
    #                 self.messages.append(AssistantMessage(content))
    #                 self.append(self.assistant_message(content))
    #             case "tool":
    #                 self.tool_demo(template_stem)

    def user_message(self, content : str):
        return {"role": "user", "content": content}

    def assistant_message(self, content : str):
        return {"role": "assistant", "content": content}

    def system_message(self, content : str):
        return {"role": "system", "content": content}
    
    def hint_message(self, content : str):
        return {"role": "user", "content": content}

    def tool_results_message(self, tool_results : List[ToolResult]):
        return self.user_message(self.formatter.tool_result_prompt(tool_results))

    def tool_calls_message(self, tool_calls : List[ToolCall]):
        return self.assistant_message(self.formatter.tool_call_prompt(tool_calls))

    def call_model_internal(self, start_seqs, stop_seqs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.outgoing_messages,
            temperature=self.args.temperature,
            max_tokens=1024,
            stop=stop_seqs,
            # frequency_penalty=-0.2,
            # Not supported in OpenAI module but VLLM supports it
            extra_body={'repetition_penalty': 1.0},
        )
        # Check if the model wants to run more tools and add the stop sequence
        if response.choices[0].finish_reason == "stop" and any(s in response.choices[0].message.content for s in start_seqs):
            # Add the stop sequence to the content
            response.choices[0].message.content += "\n" + stop_seqs[0] + "\n"
            has_tool_calls = True
        else:
            has_tool_calls = False
        message = response.choices[0].message
        content = message.content
        return response, content, message, has_tool_calls

    # TODO: make generation parameters configurable
    def call_model(self):
        # Get the delimiters from the formatter
        start_seqs, stop_seqs = self.formatter.get_delimiters()
        if self.quirks.augment_stop_sequences:
            stop_seqs = self.quirks.augment_stop_sequences(stop_seqs)
        if self.quirks.augment_start_sequences:
            start_seqs = self.quirks.augment_start_sequences(start_seqs)

        # Make the actual call to the LLM
        original_response, original_content, message, has_tool_calls = self.call_model_internal(start_seqs, stop_seqs)

        # Some models consistently mess up their output in a predictable and fixable way;
        # apply a fix if one is available.
        if self.quirks.clean_content:
            fixed_content = self.quirks.clean_content(original_content)
        else:
            fixed_content = original_content

        # Extract out the content (as opposed to the tool calls)
        extracted_content = self.formatter.extract_content(fixed_content)

        # Add the cleaned message to the log since the next part may fail
        self.append(message)

        if has_tool_calls:
            # Extract tool calls (but don't parse yet)
            try:
                tool_calls = self.formatter.extract_tool_calls(fixed_content)
                self.messages.append(UnparsedToolCalls(original_response, tool_calls, extracted_content))
                self.last_tool_calls = tool_calls
            except Exception as e:
                estr = f'{type(e).__name__}: {e}'
                status.debug_message(f"Error extracting tool calls: {estr}")
                tool_calls = []
                self.last_tool_calls = []
                self.messages.append(ErrorToolCalls(original_response, estr, extracted_content))
                self.append(
                    self.tool_results_message([
                        ToolResult(
                            name="[error]",
                            id='[none]',
                            result = {"error": f"Error extracting tool calls: {estr}"},
                        )
                    ])
                )
        else:
            self.last_tool_calls = []
            self.messages.append(AssistantMessage(extracted_content, original_response))
        return message, extracted_content

    def parse_tool_arguments(self, tool: Tool, tool_call: ToolCall) -> Tuple[bool, ToolCall | ToolResult]:
        # Don't need to parse if the arguments are already parsed;
        # this can happen if the tool call was created with parsed arguments
        if tool_call.parsed_arguments:
            return True, tool_call
        try:
            parsed_tc = self.formatter.extract_params(tool, tool_call)
            # Upgrade in-place so we get the parsed version in the log
            tool_call.parsed_arguments = parsed_tc.parsed_arguments
            return True, tool_call
        except ValueError as e:
            msg = f"{type(e).__name__} extracting parameters for {tool_call.name}: {e}"
            status.debug_message(msg)
            return False, tool_call.error(msg)

    def send(self, message : str) -> Tuple[Optional[str],bool]:
        if message:
            self.append(self.user_message(message))
        _, content = self.call_model()
        return content, self.last_tool_calls, 0

    def append(self, message : Union[dict,ChatCompletionMessage,List[ToolResult]]):
        if isinstance(message, dict):
            conv_message = message
        elif isinstance(message, ChatCompletionMessage):
            conv_message = message
        elif isinstance(message, list) and isinstance(message[0], ToolResult):
            conv_message = self.tool_results_message(message)
        else:
            raise ValueError(f"Unknown message type: {type(message)}")
        # Save the message to the log we pass back to the model
        self.outgoing_messages.append(conv_message)
        # self.messages.append(UserMessage(message))

    def get_system_message(self):
        return self.system_message_content
