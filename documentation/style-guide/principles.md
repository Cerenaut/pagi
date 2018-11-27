# Project AGI Coding Principles

## Introduction
As with PEP8, our guiding principle is code readability. Much more time will be spent reading than writing the code. Style and 
consistency are the key, but not the be-all and end-all. It’s summed up nicely in PEP8 here.

The primary language is Python and we adopt PEP8 as the primary style guide. We also adopt the TensorFlow style guide. 
The main features is 2 space indentation instead of 4.

Some additional Principles and Conventions not covered explicitly in PEP8 or Tensorflow styleguide are detailed below.

## Coding Principles
- Avoid losing ideas in the code
    - TODOs and BUGs should be a last resort
    - Do it straight away or create a card in Trello
    - If you do add TODO or BUG, create a card in Trello as well

- Commented out code
    - Remove unused code, it gets stale very quickly and creates clutter reducing readability. It will be available in version control. 
    - Unused code covers
    - long sections of dead code
    - long-dead code, and 
    - code that doesn’t get executed anymore

- Readability over brevity 
    - Avoid complicated one line compound statements
    - Avoid single letter or very brief variable names

- Documentation
    - Every class should have at least a line of documentation describing its purpose and how it should be used. 
    - The same goes for method names that do anything non-trivial.

- General
    - Naming: Lack of meaningful name suggests the design is wrong. Words like ‘Manager’, suggest the author doesn’t yet know what the 
    class is doing or what it means, and it could do with a re-design or rethink. Do not allow meaning-drift: Tight names ensure that 
    concept A doesn’t become concept B, or some sort of A/B hybrid. See: https://en.wikipedia.org/wiki/God_object “Do not create god 
    classes/objects in your system. Be very suspicious of an abstraction whose name contains Driver, Manager, System, or Subsystem.”
    - Methods: Method names should be descriptive (like variable names). Methods are easiest to write and maintain if they are short 
    and single purpose. If they extend beyond one screen, think about how you can break them up into shorter units.
    - Composition over inheritance: Results in cleaner code, less coupling, easier to maintain.
    - Class Construction: All necessary setup is done in the initialiser unless otherwise specified.
    - Instance variables: Use one underscore e.g. _var for variables that are intended to be private. 

## Coding Style Guide
- Always use brackets to show precedence, otherwise the reader needs to lookup the operator precedence table. Even if redundant it tells 
us what you intended the logic to be, which might not be what it actually does.
- Formatting.
    - Replace tabs with spaces
    - Consistent quote usage: always use single quotes, unless there’s an exception outlined here such as docstrings, or if string 
    contains single quotes
    - Compound conditionals should be set out one condition per line. This allows immediate visibility of the different components, 
    and the boolean operators between them.
	  ```if (sparsea and
           sparseb):```
           
## Conventions

### Variable Naming
- Avoid abbreviations
    - It takes longer to look up the name of a variable in the code than to auto-complete typing it.
- Give clear and descriptive names for variables and functions.
    - Don’t use very short, non standard abbreviations. They are hard for others to understand, and don’t give you information about the 
    meaning of the variable. For example: `VeryComplicatedClassifier vcc;`
    - Note that something like ‘i’ in a loop is standard, and well understood, so that is fine.
    - Unless it is a descriptive name for that context, don’t use a name for a variable that is the name of the Type (or an abbreviation 
    of the Type). It doesn’t add any information, and doesn’t describe anything about the variable. For example: 
    `AbstractPair abstract_pair;`

#### Tensorflow Naming
If we have a matrix xxxx which is needed across multiple different sessions, naming can get confusing depending whether it is a 
placeholder, numpy array, tensor graph op or output.

Convention is to name:
xxxx: Numpy ndarray (visible to python)
xxxx_pl: Placeholder to input xxxx to graph for update
xxxx_op: Graph op that produces updated xxxx.

Lifecycle of a matrix is then:
numpy --> placeholder --> op --> numpy

As code:
```
xxxx = …
feed_dict {
  xxxx_pl: xxxx,
…
op_list = [ …, xxxx_op, … ]
…, xxxx, … = sess.run( op_list )
```

### Exception Handling
Catch specific exceptions that are predicted and can be handled. Otherwise, it’s preferable to quit execution and re-run or debug the issue. Adding more specific info before the exception is printed is helpful, though note that stderr should be used or the extra info may not be printed before the process terminates.

### Console Logging Policy
We use the built in Python logging tools.
