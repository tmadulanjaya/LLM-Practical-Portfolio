# Test Conversation

Run these in the n8n chat, in order. Steps marked (approval) should first ask
"Shall I go ahead?" and only act after you reply "yes".

| # | Message | Tool expected | Approval |
|---|---------|---------------|----------|
| 1 | What are the core subjects in the BSc Computer Science program? | search_university_info | No |
| 2 | Am I free tomorrow afternoon? | get_events | No |
| 3 | Book a meeting with my advisor tomorrow at 2 PM. | create_event | Yes |
| 4 | Show the record of student S001. | get_student | No |
| 5 | Change the advisor of S001 to Dr. Fernando. | update_student | Yes |
| 6 | Write a short confirmation letter for S001. | create_doc, update_doc | Yes |
| 7 | Email the letter link to my address. | send_email | Yes |

Also try a question the handbook cannot answer (for example "What is the
cafeteria menu today?") and check that the assistant says it could not find it.
