""" Code reused from https://github.com/keras-team/keras/blob/master/docs/autogen.py """
from __future__ import print_function
from __future__ import unicode_literals

import re
import inspect
import os
import shutil

import pyshac
import pyshac.config.hyperparameters as hp
import pyshac.config.data as data
import pyshac.config.callbacks as callbacks
import pyshac.core.engine as optimizer
import pyshac.core.managed.tf_engine as tf_optimizer
import pyshac.core.managed.keras_engine as keras_optimizer
import pyshac.core.managed.torch_engine as torch_optimizer
import pyshac.utils.vis_utils as vis_utils

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding('utf8')


EXCLUDE = {

}


# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = [
    {
        'page': 'config/hyperparameters.md',
        'classes': [
            (hp.AbstractHyperParameter, '*'),
            (hp.AbstractContinuousHyperParameter, '*'),
            (hp.AbstractMultiContinuousHyperParameter, '*'),
            (pyshac.DiscreteHyperParameter, '*'),
            (pyshac.UniformContinuousHyperParameter, '*'),
            (pyshac.NormalContinuousHyperParameter, '*'),
            (pyshac.MultiDiscreteHyperParameter, '*'),
            (pyshac.MultiUniformContinuousHyperParameter, '*'),
            (pyshac.MultiDiscreteHyperParameter, '*'),
            (pyshac.HyperParameterList, [
                pyshac.HyperParameterList.add_hyper_parameter,
                pyshac.HyperParameterList.remove_hyper_parameter,
                pyshac.HyperParameterList.sample,
                pyshac.HyperParameterList.encode,
                pyshac.HyperParameterList.decode,
                pyshac.HyperParameterList.get_config,
                pyshac.HyperParameterList.load_from_config,
                pyshac.HyperParameterList.get_parameter_names,
                pyshac.HyperParameterList.set_seed,
            ]),
        ],
        'functions': [
            hp.get_parameter,
            hp.set_custom_parameter_class,
        ],
    },
    {
        'page': 'config/data.md',
        'classes': [
            (data.Dataset, '*'),
        ],
        'functions': [
            data.flatten_parameters,
        ]
    },
    {
        'page': 'config/callbacks.md',
        'classes': [
            (callbacks.Callback, '*'),
            (callbacks.History, []),
            (callbacks.CSVLogger, []),
        ],
        'functions': [
            callbacks.get_history,
        ]
    },
    {
        'page': 'core/engine.md',
        'classes': [
            (optimizer.SHAC, [
                optimizer.SHAC.fit,
                optimizer.SHAC.fit_dataset,
                optimizer.SHAC.predict,
                optimizer.SHAC.save_data,
                optimizer.SHAC.restore_data,
                optimizer.SHAC.set_num_parallel_generators,
                optimizer.SHAC.set_num_parallel_evaluators,
                optimizer.SHAC.parallel_evaluators,
                optimizer.SHAC.concurrent_evaluators,
                optimizer.SHAC.set_seed,
                optimizer.SHAC.as_deterministic,
            ]),
        ],
    },
    {
        'page': 'core/tf_engine.md',
        'classes': [
            (tf_optimizer.TensorflowSHAC, [
                tf_optimizer.TensorflowSHAC.fit,
                tf_optimizer.TensorflowSHAC.fit_dataset,
                tf_optimizer.TensorflowSHAC.predict,
                tf_optimizer.TensorflowSHAC.save_data,
                tf_optimizer.TensorflowSHAC.restore_data,
                tf_optimizer.TensorflowSHAC.set_num_parallel_generators,
                tf_optimizer.TensorflowSHAC.set_num_parallel_evaluators,
                tf_optimizer.TensorflowSHAC.parallel_evaluators,
                tf_optimizer.TensorflowSHAC.concurrent_evaluators,
                tf_optimizer.TensorflowSHAC.set_seed,
                tf_optimizer.TensorflowSHAC.as_deterministic,
            ]),
        ],
    },
    {
        'page': 'core/keras_engine.md',
        'classes': [
            (keras_optimizer.KerasSHAC, [
                keras_optimizer.KerasSHAC.fit,
                keras_optimizer.KerasSHAC.fit_dataset,
                keras_optimizer.KerasSHAC.predict,
                keras_optimizer.KerasSHAC.save_data,
                keras_optimizer.KerasSHAC.restore_data,
                keras_optimizer.KerasSHAC.set_num_parallel_generators,
                keras_optimizer.KerasSHAC.set_num_parallel_evaluators,
                keras_optimizer.KerasSHAC.parallel_evaluators,
                keras_optimizer.KerasSHAC.concurrent_evaluators,
                keras_optimizer.KerasSHAC.set_seed,
                keras_optimizer.KerasSHAC.as_deterministic,
            ]),
        ],
    },
    {
        'page': 'core/torch_engine.md',
        'classes': [
            (torch_optimizer.TorchSHAC, [
                torch_optimizer.TorchSHAC.fit,
                torch_optimizer.TorchSHAC.fit_dataset,
                torch_optimizer.TorchSHAC.predict,
                torch_optimizer.TorchSHAC.save_data,
                torch_optimizer.TorchSHAC.restore_data,
                torch_optimizer.TorchSHAC.set_num_parallel_generators,
                torch_optimizer.TorchSHAC.set_num_parallel_evaluators,
                torch_optimizer.TorchSHAC.parallel_evaluators,
                torch_optimizer.TorchSHAC.concurrent_evaluators,
                torch_optimizer.TorchSHAC.set_seed,
                torch_optimizer.TorchSHAC.as_deterministic,
            ]),
        ],
    },
    {
        'page': 'utils/vis_utils.md',
        'functions': [
            vis_utils.plot_dataset,
        ],
    },
    # {
    #     'page': '',
    #     'functions': [
    #
    #     ],
    # },
    # {
    #     'page': '',
    #     'functions': [
    #
    #     ],
    # },
]

# TODO: Add this
ROOT = 'http://titu1994.github.io/pyshac/'


def get_function_signature(function, method=True):
    wrapped = getattr(function, '_original_function', None)
    if wrapped is None:
        signature = inspect.getargspec(function)
    else:
        signature = inspect.getargspec(wrapped)
    defaults = signature.defaults
    if method:
        args = signature.args[1:]
    else:
        args = signature.args
    if defaults:
        kwargs = zip(args[-len(defaults):], defaults)
        args = args[:-len(defaults)]
    else:
        kwargs = []
    st = '%s.%s(' % (clean_module_name(function.__module__), function.__name__)

    for a in args:
        st += str(a) + ', '
    for a, v in kwargs:
        if isinstance(v, str):
            v = '\'' + v + '\''
        st += str(a) + '=' + str(v) + ', '
    if kwargs or args:
        signature = st[:-2] + ')'
    else:
        signature = st + ')'

    if not method:
        # Prepend the module name.
        signature = clean_module_name(function.__module__) + '.' + signature

    return post_process_signature(signature)


def get_class_signature(cls):
    try:
        class_signature = get_function_signature(cls.__init__)
        class_signature = class_signature.replace('__init__', cls.__name__)
    except (TypeError, AttributeError):
        # in case the class inherits from object and does not
        # define __init__
        class_signature = clean_module_name(cls.__module__) + '.' + cls.__name__ + '()'
    return post_process_signature(class_signature)


def post_process_signature(signature):
    # parts = re.split(r'\.(?!\d)', signature)
    # if len(parts) >= 4:
    #     if parts[1] == 'layers':
    #         signature = 'keras.layers.' + '.'.join(parts[3:])
    #     if parts[1] == 'utils':
    #         signature = 'keras.utils.' + '.'.join(parts[3:])
    #     if parts[1] == 'backend':
    #         signature = 'keras.backend.' + '.'.join(parts[3:])
    return signature


def clean_module_name(name):
    # if name.startswith('keras_applications'):
    #     name = name.replace('keras_applications', 'keras.applications')
    # if name.startswith('keras_preprocessing'):
    #     name = name.replace('keras_preprocessing', 'keras.preprocessing')
    # assert name[:6] == 'keras.', 'Invalid module name: %s' % name
    return name


def class_to_docs_link(cls):
    module_name = clean_module_name(cls.__module__)
    module_name = module_name[6:]
    link = ROOT + module_name.replace('.', '/') + '#' + cls.__name__.lower()
    return link


def class_to_source_link(cls):
    module_name = clean_module_name(cls.__module__)
    path = module_name.replace('.', '/')
    path += '.py'
    line = inspect.getsourcelines(cls)[-1]
    link = ('https://github.com/titu1994/'
            'pyshac/blob/master/' + path + '#L' + str(line))
    return '[[source]](' + link + ')'


def code_snippet(snippet):
    result = '```python\n'
    result += snippet + '\n'
    result += '```\n'
    return result


def count_leading_spaces(s):
    ws = re.search(r'\S', s)
    if ws:
        return ws.start()
    else:
        return 0


# keras definition
# def process_list_block(docstring, starting_point, leading_spaces, marker):
#     ending_point = docstring.find('\n\n', starting_point)
#     block = docstring[starting_point:None if ending_point == -1 else ending_point - 1]
#     # Place marker for later reinjection.
#     docstring = docstring.replace(block, marker)
#     lines = block.split('\n')
#     # Remove the computed number of leading white spaces from each line.
#     lines = [re.sub('^' + ' ' * leading_spaces, '', line) for line in lines]
#     # Usually lines have at least 4 additional leading spaces.
#     # These have to be removed, but first the list roots have to be detected.
#     top_level_regex = r'^    ([^\s\\\(]+):(.*)'
#     top_level_replacement = r'- __\1__:\2'
#     lines = [re.sub(top_level_regex, top_level_replacement, line) for line in lines]
#     # All the other lines get simply the 4 leading space (if present) removed
#     lines = [re.sub(r'^    ', '', line) for line in lines]
#     # Fix text lines after lists
#     indent = 0
#     text_block = False
#     for i in range(len(lines)):
#         line = lines[i]
#         spaces = re.search(r'\S', line)
#         if spaces:
#             # If it is a list element
#             if line[spaces.start()] == '-':
#                 indent = spaces.start() + 1
#                 if text_block:
#                     text_block = False
#                     lines[i] = '\n' + line
#             elif spaces.start() < indent:
#                 text_block = True
#                 indent = spaces.start()
#                 lines[i] = '\n' + line
#         else:
#             text_block = False
#             indent = 0
#     block = '\n'.join(lines)
#     return docstring, block


def process_list_block(docstring, starting_point, leading_spaces, marker):
    ending_point = docstring.find('\n\n', starting_point)
    block = docstring[starting_point:None if ending_point == -1 else ending_point - 1]
    # Place marker for later reinjection.
    docstring = docstring.replace(block, marker)
    lines = block.split('\n')
    # Remove the computed number of leading white spaces from each line.
    lines = [re.sub('^' + ' ' * leading_spaces, '', line) for line in lines]
    # Usually lines have at least 4 additional leading spaces.
    # These have to be removed, but first the list roots have to be detected.
    top_level_regex = r'^    ([^\s\\\(]+):(.*)'
    top_level_replacement = r'- __\1__:\2'
    lines = [re.sub(top_level_regex, top_level_replacement, line) for line in lines]
    # All the other lines get simply the 4 leading space (if present) removed
    lines = [re.sub(r'^    ', '', line) for line in lines]
    # Fix text lines after lists
    indent = 0
    text_block = False
    for i in range(len(lines)):
        line = lines[i]

        spaces = re.search(r'\S', line)
        parameter = re.search(r'.+?([a-zA-Z])\):', line)

        if spaces:
            # If it is a list element
            if line[spaces.start()] == '-':
                indent = spaces.start() + 1
                if text_block:
                    text_block = False
                    lines[i] = '\n' + line
            elif spaces.start() < indent:
                text_block = True
                indent = spaces.start()
                lines[i] = '\n' + line
        else:
            text_block = False
            indent = 0

        if parameter:
            # Add a new bullet point, with bold
            line = re.sub(r'(.+?([a-zA-Z])\):)', r'- **\1**', line)
            lines[i] = line

    block = '\n'.join(lines)

    return docstring, block


def process_docstring(docstring):
    # First, extract code blocks and process them.
    code_blocks = []
    if '```' in docstring:
        tmp = docstring[:]
        while '```' in tmp:
            tmp = tmp[tmp.find('```'):]
            index = tmp[3:].find('```') + 6
            snippet = tmp[:index]
            # Place marker in docstring for later reinjection.
            docstring = docstring.replace(
                snippet, '$CODE_BLOCK_%d' % len(code_blocks))
            snippet_lines = snippet.split('\n')
            # Remove leading spaces.
            num_leading_spaces = snippet_lines[-1].find('`')
            snippet_lines = ([snippet_lines[0]] +
                             [line[num_leading_spaces:]
                             for line in snippet_lines[1:]])
            # Most code snippets have 3 or 4 more leading spaces
            # on inner lines, but not all. Remove them.
            inner_lines = snippet_lines[1:-1]
            leading_spaces = None
            for line in inner_lines:
                if not line or line[0] == '\n':
                    continue
                spaces = count_leading_spaces(line)
                if leading_spaces is None:
                    leading_spaces = spaces
                if spaces < leading_spaces:
                    leading_spaces = spaces
            if leading_spaces:
                snippet_lines = ([snippet_lines[0]] +
                                 [line[leading_spaces:]
                                  for line in snippet_lines[1:-1]] +
                                 [snippet_lines[-1]])
            snippet = '\n'.join(snippet_lines)
            code_blocks.append(snippet)
            tmp = tmp[index:]

    # Format docstring lists.
    section_regex = r'\n( +)# (.*)\n'
    section_idx = re.search(section_regex, docstring)
    shift = 0
    sections = {}
    while section_idx and section_idx.group(2):
        anchor = section_idx.group(2)
        leading_spaces = len(section_idx.group(1))
        shift += section_idx.end()
        marker = '$' + anchor.replace(' ', '_') + '$'
        docstring, content = process_list_block(docstring,
                                                shift,
                                                leading_spaces,
                                                marker)
        sections[marker] = content
        section_idx = re.search(section_regex, docstring[shift:])

    # Format docstring section titles.
    docstring = re.sub(r'\n(\s+)# (.*)\n',
                       r'\n\1__\2__\n\n',
                       docstring)

    # Strip all remaining leading spaces.
    lines = docstring.split('\n')
    docstring = '\n'.join([line.lstrip(' ') for line in lines])

    # Reinject list blocks.
    for marker, content in sections.items():
        docstring = docstring.replace(marker, content)

    # Reinject code blocks.
    for i, code_block in enumerate(code_blocks):
        docstring = docstring.replace(
            '$CODE_BLOCK_%d' % i, code_block)
    return docstring


print('Cleaning up existing sources directory.')
if os.path.exists('sources'):
    shutil.rmtree('sources')

print('Populating sources directory with templates.')
for subdir, dirs, fnames in os.walk('templates'):
    for fname in fnames:
        new_subdir = subdir.replace('templates', 'sources')
        if not os.path.exists(new_subdir):
            os.makedirs(new_subdir)
        if fname[-3:] == '.md':
            fpath = os.path.join(subdir, fname)
            new_fpath = fpath.replace('templates', 'sources')
            shutil.copy(fpath, new_fpath)


def read_file(path):
    with open(path) as f:
        return f.read()


def collect_class_methods(cls, methods):
    if isinstance(methods, (list, tuple)):
        return [getattr(cls, m) if isinstance(m, str) else m for m in methods]
    methods = []
    for _, method in inspect.getmembers(cls, predicate=inspect.isroutine):
        if method.__name__[0] == '_' or method.__name__ in EXCLUDE:
            continue
        methods.append(method)
    return methods


def render_function(function, method=True):
    subblocks = []
    signature = get_function_signature(function, method=method)
    signature = signature.replace(function.__module__ + '.', '')
    subblocks.append('### ' + function.__name__ + '\n')
    subblocks.append(code_snippet(signature))
    docstring = function.__doc__
    if docstring:
        subblocks.append(process_docstring(docstring))
    return '\n\n'.join(subblocks)


def read_page_data(page_data, type):
    assert type in ['classes', 'functions']
    data = page_data.get(type, [])
    for module in page_data.get('all_module_{}'.format(type), []):
        module_data = []
        for name in dir(module):
            if name[0] == '_' or name in EXCLUDE:
                continue
            module_member = getattr(module, name)
            if (inspect.isclass(module_member) and type == 'classes' or
               inspect.isfunction(module_member) and type == 'functions'):
                instance = module_member
                if module.__name__ in instance.__module__:
                    if instance not in module_data:
                        module_data.append(instance)
        module_data.sort(key=lambda x: id(x))
        data += module_data
    return data


if __name__ == '__main__':
    readme = read_file('../README.md')
    index = read_file('templates/index.md')
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    with open('sources/index.md', 'w') as f:
        f.write(index)

    print('Generating docs for PySHAC %s.' % pyshac.__version__)
    for page_data in PAGES:
        classes = read_page_data(page_data, 'classes')

        blocks = []
        for element in classes:
            if not isinstance(element, (list, tuple)):
                element = (element, [])
            cls = element[0]
            subblocks = []
            signature = get_class_signature(cls)
            subblocks.append('<span style="float:right;">' +
                             class_to_source_link(cls) + '</span>')
            if element[1]:
                subblocks.append('## [' + cls.__name__ + '](#%s)\n' % (cls.__name__.lower()))
            else:
                subblocks.append('### ' + cls.__name__ + '\n')
            subblocks.append(code_snippet(signature))
            docstring = cls.__doc__
            if docstring:
                subblocks.append(process_docstring(docstring))
            methods = collect_class_methods(cls, element[1])
            if methods:
                subblocks.append('\n---')  # subblocks.append('\n---')
                subblocks.append('## ' + cls.__name__ + ' methods\n')
                subblocks.append('\n---\n'.join(
                    [render_function(method, method=True) for method in methods]))
            blocks.append('\n'.join(subblocks))

        functions = read_page_data(page_data, 'functions')

        for function in functions:
            blocks.append(render_function(function, method=False))

        if not blocks:
            raise RuntimeError('Found no content for page ' +
                               page_data['page'])

        mkdown = '\n----\n\n'.join(blocks)
        # save module page.
        # Either insert content into existing page,
        # or create page otherwise
        page_name = page_data['page']
        path = os.path.join('sources', page_name)
        if os.path.exists(path):
            template = read_file(path)
            assert '{{autogenerated}}' in template, ('Template found for ' + path +
                                                     ' but missing {{autogenerated}} tag.')
            mkdown = template.replace('{{autogenerated}}', mkdown)
            print('...inserting autogenerated content into template:', path)
        else:
            print('...creating new page with autogenerated content:', path)
        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        with open(path, 'w') as f:
            f.write(mkdown)
