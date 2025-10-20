RAG_PROMPT_TEMPLATE = """
{system_instructions}

INSTRUCTIONS FOR SPECIAL CASES:
- If the USER QUESTION is a general greeting or polite expression (e.g., "hi", "hello", "good afternoon", "thank you"), respond naturally and politely, without referencing CONTEXT.
- Otherwise, follow the instructions below.

CONTEXT: 
{context}

PATIENT:
{patient_info}

USER QUESTION:
{user_question}

INSTRUCTIONS FOR HEALING ANSWERS:
- Use ONLY facts and procedures found in CONTEXT.
- When asked to produce a healing protocol, produce it in the exact structured format shown in the example below.
- If multiple candidate methods are found, prefer those explicitly mentioning color/prana matching to disease. 
- If context doesn't have guidance, say "No established protocol in the documents" and suggest a gentle diagnostic protocol to determine color prana.

OUTPUT FORMAT EXAMPLE:
Standard Psycho Therapy as per track (basic chakra sudhi)

General cleansing with LWG

Affected part of left leg:
Localize with LWB 
Soak with LWG, LWO
Wait
Clean
Again localize with LWB 
Energize with LWG, LWB, LWV

Basic chakra: clean with LWG, LWO. Energize with White

Sex chakra: clean with LWG, LWO. Energize with White 

All minor chakras of leg: clean with LWG, LWO. Energize with White

Now, produce the answer for the user question using the CONTEXT above. 
If you reference document text, include the Source: <filename> lines inline (short). Be concise.
"""
