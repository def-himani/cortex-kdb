## Issue 002: Ollama Context Window Mismatch

Status: Resolved

Problem:
Ollama allocated 4096 tokens while kdb.toml specified 16384.

Root Cause:
The intended context configuration was not applied to
the running Ollama model.

Solution:
Created models/Modelfile with PARAMETER num_ctx 16384
and registered cortex-gemma4:16k.

Verification:
ollama ps confirmed 16384 context and 100% GPU offloading.


## Issue 003: Pydantic AI RunUsage TypeError

Status: Resolved

Problem:
TypeError: 'RunUsage' object is not callable.

Solution:
Replaced result.usage() with result.usage.

Verification:
Token usage was successfully reported by check_model.py.


## Issue 004: Git Branch Synchronization

Status: Resolved

Problem:
Project documentation was missing from feature/model-setup.

Root Cause:
The project-foundation pull request was merged into main
instead of develop.

Solution:
Synchronized develop with main and merged develop
into feature/model-setup.

Prevention:
Always verify that feature pull requests target develop.