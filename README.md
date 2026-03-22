Is a fork of https://github.com/composable-models/llm_multiagent_debate

The purpose of this fork is to extend the original MAS with CacheSaver.
https://github.com/au-clan/cachesaver

### How to set up

You need to have uv install, which you can do with pip. 

Then run the following commands:
```
clone https://github.com/Rasmuskjriis/llm_multiagent_debate_cachesaver.git
cd llm_multiagent_debate_cachesaver
uv sync
```

After which you can run our code.

You can try the following command, which test if 2 agents can solve a random math problem over 3 rounds.

```
uv run math/gen_math.py
```