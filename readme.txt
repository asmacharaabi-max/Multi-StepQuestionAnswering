LCEL Multi-Step Question Answering App:
This application is a Flask-based AI system built using LangChain Expression Language (LCEL).
Its main goal is to demonstrate how complex user questions can be processed through a structured AI pipeline.

The app uses a large language model (LLM) to break down a question into smaller sub-questions, answer them in parallel, and then generate a final synthesized response.

The system follows a 3-step pipeline:

1. Question Decomposition

The user input is analyzed and broken down into up to 3 sub-questions using an LLM.

2. Parallel Answering

Each sub-question is sent to the LLM independently using RunnableParallel, allowing simultaneous execution.

3. Final Synthesis

All sub-question answers are combined and passed to a final LLM step to generate a concise, unified response.