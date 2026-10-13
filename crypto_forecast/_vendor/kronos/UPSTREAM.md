# Kronos upstream source

Source: https://github.com/shiyu-coder/Kronos

Revision: `67b630e67f6a18c9e9be918d9b4337c960db1e9a`

The upstream `model/kronos.py`, `model/module.py`, and `model/__init__.py`
are included under the accompanying MIT license. The only packaging change
replaces `sys.path.append` and `from model.module import *` with a relative
`from .module import *`, preventing collisions with unrelated `model` packages.

Pretrained weights are downloaded separately from the official Hugging Face
repositories `NeoQuasar/Kronos-small` and `NeoQuasar/Kronos-Tokenizer-base`.
