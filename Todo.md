# Agenda

## math
 [X] Find a list of parameters to test for like tokens, time, accuracy etc.
 [] Figure out how to measure these parameters in our test and represent them.
 [X] Make a test file for gen_math.py that tests with/without CacheSaver.
 [X] Rework gen_math.py to take hyperparameters such as agents, rounds, with/without CacheSaver etc.
 [] Maybe test if running multiple instances of Ollama improves performance.

 [] Add temperature for randomness in the LLM response.
 [] Make agents more critically thinking.

## gsm


## biography


## mmlu



[] Try running Ollama with LocalModel instead of OnlineModel to see if that fixes the batching issue and maybe also the terrible tokenizer





[{'role': 'user', 'content': 'Give a bullet point biography of Darrell Long highlighting their contributions and achievements as a computer scientist, with each fact separated with a new line character. '},

{'role': 'assistant', 'content': 

"Here's a bullet point biography of Darrell Long:\n\n*

Darrell Long is a renowned computer scientist and professor in the Department of Computer Science at the University of California, Santa Cruz.\n\n*
 
He received his Bachelor's degree in Computer Science from the University of California, Berkeley, and his Ph.D. in Computer Science from the University of California, Berkeley.\n\n*

Long's research focuses on computer systems, particularly in the areas of operating systems, file systems, and storage systems.\n\n*

He has made significant contributions to the development of several file systems, including the Sprite file system, which was one of the first network file systems.\n\n*

Long has also worked on data archival and storage systems, and his work on the LOCKSS (Lots of Copies Keep Stuff Safe) digital preservation system has been widely adopted by libraries and archives.\n\n*

He has published numerous papers and articles in top-tier conferences and journals, including the ACM Transactions on Computer Systems and the IEEE Transactions on Computers.\n\n* Long has received several awards and honors for his contributions to computer science, including the IEEE Computer Society's Technical Committee on Operating Systems Outstanding Achievement Award.\n\n*

He has served on the program committees of several major conferences, including the ACM Symposium on Operating Systems Principles and the IEEE International Conference on Distributed Computing Systems.\n\n*

Long has also been a fellow of the American Association for the Advancement of Science and a member of the IEEE Computer Society's Hall of Fame."}]]