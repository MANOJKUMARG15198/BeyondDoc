# My decisions in this project

I was involved in this project end to end: the planning, the decision making, the next steps and the choice of
every component.

## Model, concept and search method

I decided the models, the overall RAG concept and the vector database search methods from my previous working
experience with RAG systems.

## Trial and error

I made the detailed choices through trial and error on sample questions and answers: chunking, embedding, retrieval
and the answer style.

- **Chunking:** I chose the chunking method and the chunk sizes myself. I count chunk tokens with BERT uncased,
  chosen for this medical data.
- **Output structure:** I shaped how the answer is delivered through the prompt, testing it on different case
  scenarios with the data (for example, questions with no simple yes or no, step-by-step dosing, and questions the
  documents only partly cover).

## User interface

I gave the instructions for the UI development and used different kinds of models for the UI design and the
naming. The app is named **Beyond Doc**; its design started from a Claude Design handoff, which I adapted to our
documents (for example, no patient records and no evidence grades).

## Accuracy first

I spent most of the time on accuracy, so the answers meet the expectations for medical data: answers come only from
the documents, cite their sources, and say clearly when the documents do not cover a question.

## Engineering and AI tools

I used different AI tools to review the code and to optimize the evaluation code, and every change went through my
approval first.

- **Development:** Claude Code, with the Opus 5.5 model.
- **Code review:** ChatGPT, OpenAI Sol models and Kimi K3.
