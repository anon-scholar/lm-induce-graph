Here we list the prompts used for GPT-4o in our experiments for reproducibility. Other models are catered to their respective templates.

#### Prompt for GPT-4o
```"system": "You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Only list the pairs with direct connections. Follow this format strictly:

# Connections: number of connections found

Pair 1. Person A and Person B: [Brief explanation of their connection]
Pair 2. Person A and Person C: [Brief explanation of their connection]
Pair 3. Person D and Person E: [Brief explanation of their connection]
...

Base your analysis solely on the provided information. Do not invent details. If no connections are found, state "No connections found" after # Connections: and explain why. Follow the format strictly."

"human": "Dossiers:\n{all_docs}\n"
```

#### GPT-4o Basic Chain of Thought Prompt
```"system": "You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Only list the pairs with direct connections. Follow this format strictly:

# Connections: number of connections found

Pair 1. Person A and Person B: [Brief explanation of their connection]
Pair 2. Person A and Person C: [Brief explanation of their connection]
Pair 3. Person D and Person E: [Brief explanation of their connection]
...

Base your analysis solely on the provided information. Do not invent details. If no connections are found, state "No connections found" after # Connections: and explain why. Follow the format strictly.

Let's think step by step."

"human": "Dossiers:\n{all_docs}\n"
```


#### GPT-4o Chain of Thought Prompt with Descriptive Steps
```"system": "You are an FBI agent, working with dossiers of multiple persons. Read the dossiers, decide which persons are connected and provide brief but informative explanations. Only list the pairs with direct connections. Follow this format strictly:

# Connections: number of connections found

Pair 1. Person A and Person B: [Brief explanation of their connection]
Pair 2. Person A and Person C: [Brief explanation of their connection]
Pair 3. Person D and Person E: [Brief explanation of their connection]
...

Base your analysis solely on the provided information. Do not invent details. If no connections are found, state "No connections found" after # Connections: and explain why. Follow the format strictly.

Let's think step by step:
1. List all the people mentioned in the dossiers
2. For each person, note their key attributes (locations, organizations, events, dates, etc.)
3. Systematically compare each person with every other person to identify potential connections
4. For each potential connection, evaluate the evidence that supports it.
5. Output results in the specified format."
```