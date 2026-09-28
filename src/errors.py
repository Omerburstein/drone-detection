"""The one exception a library module raises when the *input* is wrong.

Modules under `src/` used to call `sys.exit` on bad input, on the reasoning that
a mistyped path is a user error rather than an exceptional condition. That is
true of the message and false of the mechanism: it also killed the interpreter
for anyone calling the same function from a notebook or a test, which is exactly
the use this project's own guidance promises ("constructible directly, so
inference is callable from a notebook or test without a command line").

So the library raises `UsageError` and each CLI's `main()` turns it back into
`sys.exit(str(exc))`. The exit code and the message the user sees are unchanged;
what changes is that the decision to end the process now belongs to the entry
point instead of to a function four frames down.

`UsageError` is for input a *user* got wrong and can fix -- a missing file, a
malformed flag, a label file that does not parse. A bug in our own code should
still raise the ordinary exception, loudly and with a traceback.

Imports nothing, so any module may raise it without acquiring a dependency.
"""

from __future__ import annotations


class UsageError(Exception):
    """Bad input the user can correct, reported without a traceback."""
