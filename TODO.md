(blocks + FTS + vector index) - what is FTS (full text search?)
[ ] Clean or remve digest prompt.
[ ] synthesis_prompt.md needs to be reviewed, right now the digest and digital garden features are separate. Those features need to be merged more elegantly. Ideally, I want to have one platform that does several things, note taking (LLM helps with grooming, note merging, synthesis, etc.), learning (I put my raw thoughts and LLM connects them, find flaws in them, etc.), I track my tasks, projects and goals (LLM helps me to form better tasks, split them, etc), journaling (LLM helps me finding next move, exerice, thought reframing)
[ ] https://vscode.dev/github/synerdjin/Dendr/blob/main/src/dendr/templates/synthesis_prompt.md#L88 Is Claude able to distinguish when to use tools?
[ ] https://vscode.dev/github/synerdjin/Dendr/blob/main/src/dendr/templates/synthesis_prompt.md#L98 This is not a desired behavior, I want data to be searched more often, like to see if claim is progressing, etc. 
[ ] https://vscode.dev/github/synerdjin/Dendr/blob/main/scripts/update.sh#L22 remove if not needed.
[ ] Research if better quantization needed for embedding gemma.
[ ] version is not bamping in __init__.py, do we need version there.
[ ] Clean Pre-v8 stuff
[ ] https://vscode.dev/github/synerdjin/Dendr/blob/main/src/dendr/config.py#L21 Would it make sense to pull the version from dendr-models.yaml?
