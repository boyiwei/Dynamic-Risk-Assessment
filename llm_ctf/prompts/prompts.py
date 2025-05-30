#!/usr/bin/env python3

from collections import defaultdict
from typing import List, TYPE_CHECKING
from jinja2 import Environment, PackageLoader, StrictUndefined
from jinja2.exceptions import TemplateNotFound
from jinja2 import Template

from nyuctf.challenge import CTFChallenge
from ..tools import Tool, ToolCall, ToolResult

if TYPE_CHECKING:
    from ..formatters import Formatter
import os

DEFAULT_PROMPT_SET = 'default'

def blockquote(text):
    return '\n'.join([f'> {line}' for line in text.split('\n')])

class RelEnvironment(Environment):
    """Override join_path() to enable relative template paths."""
    def join_path(self, template, parent):
        return os.path.normpath(os.path.join(os.path.dirname(parent), template))

class FallbackLoader(PackageLoader):
    """A Jinja2 loader that falls back to a default collection."""
    def __init__(
            self,
            package_name: str,
            package_path: str = "templates",
            encoding: str = "utf-8",
            prompt_set: str = DEFAULT_PROMPT_SET) -> None:
        super().__init__(package_name, package_path, encoding)
        self.prompt_set = prompt_set

    def get_source(self, environment, template):
        try:
            return super().get_source(environment, os.path.normpath(f'{self.prompt_set}/{template}'))
        except TemplateNotFound:
            return super().get_source(environment, os.path.normpath(f'{DEFAULT_PROMPT_SET}/{template}'))

class PromptManager:
    def __init__(self, prompt_set=DEFAULT_PROMPT_SET, config=None):
        self.prompt_set = prompt_set
        self.env = RelEnvironment(
            loader=FallbackLoader('llm_ctf.prompts', prompt_set=prompt_set),
            autoescape=False,
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=['jinja2.ext.do'],
        )
        self.env.filters['blockquote'] = blockquote
        self.prompts = {}
        self.config = config
        self.prompt_config = {}
        self.hint_file = []
        if self.config:
            # self.prompt_config = self.config.get("prompts", None)
            self.hint_file = config.get("demostration", {}).get("hints", [])

    def render(self, name, **kwargs):
        # Overwrite prompt if the prompt in config exists
        if name not in self.prompts:
            if name == 'hint':
                self.prompts[name] = self.env.from_string(self.get_hints(self.hint_file))
            elif len(self.prompt_config.get(name, "")) > 0:
                self.prompts[name] = self.env.from_string(self.prompt_config[name])
            else:
                self.prompts[name] = self.env.get_template(f'{name}.md.jinja2')
        return self.prompts[name].render(**kwargs)

    def tool_use(
            self,
            formatter : 'Formatter',
            tools : List[Tool],
            example_call : ToolCall = None,
            **kwargs
        ):
        tool_use_example = formatter.format_tool_calls(
            [example_call] if example_call else [],
            placeholder=True,
            **kwargs,
        )
        return self.render(
            'tool_use',
            tool_list=formatter.format_tools(tools),
            tool_use_example=tool_use_example,
            formatter=formatter,
        )

    def tool_calls(
            self,
            formatter: 'Formatter',
            tool_calls: List[ToolCall],
            **kwargs):
        return self.render(
            'tool_calls',
            formatter=formatter,
            tool_calls=formatter.format_tool_calls(tool_calls),
            **kwargs,
        )

    def tool_results(
            self,
            formatter: 'Formatter',
            tool_results: List[ToolResult],
            **kwargs):
        return self.render(
            'tool_results',
            formatter=formatter,
            tool_results=formatter.format_results(tool_results),
            **kwargs
        )

    def initial_message(self, chal: CTFChallenge, **kwargs):
        return self.render('initial_message', chal=chal, **kwargs)

    def get_chal_hint(self, hint: str):
        """Get hint from the challenge dir. Return None if not present."""
        # hintpath = chal.chaldir / f"hints/{hint}.md"
        try:
            with open(hint, 'r') as f:
                return f.read()
        except FileExistsError:
            return ""
        
    def get_hints(self, hints: list, **kwargs):
        """
        Look for hints in the the challenge folder and templates folder.
        """
        if not hints:
            return ""
        msg = []
        for hint in hints:
            if ht := self.get_chal_hint(hint):
                msg.append(ht)
        hint_message = "\n\n".join(msg) if len(msg) > 0 else ""
        return hint_message

    def hints_message(self, **kwargs):
        return self.render('hint', **kwargs)

    def keep_going(self, **kwargs):
        return self.render('keep_going', **kwargs)

    def system_message(self, chal, **kwargs):
        return self.render('system', chal=chal, **kwargs)
