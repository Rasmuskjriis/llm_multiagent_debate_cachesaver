Week 11:
- We made generate_answer() and main() into asynchronous functions and used await and asyncio's gather and run functions. This way the agents run asynchronously which resulted in a performance increase from 2 min 45 sec running synchronously for 10 evaluation rounds to 2 min 9 sec running asynchronously.

- We set "OLLAMA_NUM_PARALLEL = 4" instead of 1 on our local pc, which resulted in an increase in performance from 2 min 9 sec to 1 min 47 sec. 

These two changes in terms of asynchrony resulted in about a 40% increase in performance making testing much faster and easier.