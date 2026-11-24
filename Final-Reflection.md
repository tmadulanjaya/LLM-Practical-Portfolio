# Final Reflection

## What I Learned
In these six labs I learned how computers understand and create language, step by step.

- **Lab 01 – Machine Learning:** I learned that text must be cleaned and turned into numbers before a model can use it. I used Bag of Words and Logistic Regression to tell positive and negative reviews apart, and the model reached about 82% accuracy.
- **Lab 02 – LLMs in Everyday Life:** I saw how people use LLMs in daily life, and that their answers must be checked.
- **Lab 03 – Tiny Transformer:** I built a small Transformer that learned to write simple children's stories one character at a time. Bigger models and more training gave better results.
- **Lab 04 – Fine-Tuning:** I used LoRA to teach a small model about a computer shop. It answered shop questions correctly 67% of the time, compared with 17% for the original model.
- **Lab 05 – RAG Pipeline:** I built a chatbot that answers questions from university PDF files and shows the source page.
- **Lab 06 – AI Agents:** I built agents that use tools like Calculator, Notepad, memory and Google Calendar. I also made a similar agent in n8n.

## Hardest Part
Setting things up and fixing errors was the hardest part. A copied virtual environment stopped working, and I had to create a new one. Connecting Google Calendar needed OAuth setup. Small AI models also made mistakes, such as choosing the wrong tool or inventing information.

## Favourite Lab
Lab 06 (AI Agents) was my favourite. The agent could open Calculator, remember my name and add events to my calendar. It felt like the AI was really doing work instead of only answering questions.

## What I Learned About Making AI Better
- Good data makes a big difference. The fine-tuned model improved because it was trained on shop questions.
- RAG helps the AI use real documents instead of guessing.
- Clear instructions help the AI choose the right tool.
- Testing with new questions shows the real weaknesses of a model.

## How I Will Use This
I want to build useful software for small businesses, such as a chatbot that answers customer questions from documents, or an assistant that books appointments automatically.

## Problems with AI
- **Wrong answers:** AI can make up facts with confidence (hallucination). In Lab 04, the model gave a wrong price. I must always check important answers.
- **Privacy:** Agents can see personal data like calendars and memory. I must keep passwords, API keys and token files private and never upload them to GitHub.
- **Human control:** A person should approve important actions, like creating events or sending emails.

## Conclusion
These labs showed me the full path from simple machine learning to AI agents. The main lesson is that AI is very useful, but it needs good data, clear instructions, testing and human checking.