Is a fork of https://github.com/composable-models/llm_multiagent_debate

The purpose of this fork is to extend the original MAS with CacheSaver.
https://github.com/au-clan/cachesaver

### How to set up

You need to have uv installed, which you can do with pip. 

Then run the following commands:
```
clone https://github.com/Rasmuskjriis/llm_multiagent_debate_cachesaver.git
cd llm_multiagent_debate_cachesaver
uv sync --group cachesaver
```

Then create a .env in the root folder and enter you openai key like "OPENAI_API_KEY=<your_api_key>".

You can try the following command, which test if 2 agents can solve a random math problem over 3 rounds.

```
uv run math/gen_math.py -a 2 -r 3 -p 1
```

### Rerunning our experiments
to rerun the experiment we ran as part of our project you can use the following commands.

Benchmark experiment:
```
uv run experiment/experiment.py -p 100 -m gpt-5-nano-2025-08-07
```

Hyperparameter tuning experiment:
```
uv run experiment/param_optimization_agents.py -a 4 -p 100 -m gpt-5-nano-2025-08-07
uv run experiment/param_optimization_rounds.py -r 4 -p 100 -m gpt-5-nano-2025-08-07
```