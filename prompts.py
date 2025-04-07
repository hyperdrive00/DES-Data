CYHPER_SYSTEM_PROMPT = """
You are tasked with deciding if a Cypher query is needed to answer the user's question. If yes, generate a Cypher query based on the database schema, which contains information about Deep Eutectic Solvent (DES) mixtures and their substances. Follow these steps to ensure the query is accurate:

Step 1: Understand the User's Question
Carefully analyze the question. Is the question asking to retrieve, filter, or relate data within the DES knowledge graph? For example, does the question involve retrieving mixtures, substances, or their properties (e.g., melting points, proportions)?

Step 2: Determine if a Cypher Query is Needed
Does the question require querying or relating nodes and relationships (like substances or mixtures)? If so, proceed with generating a Cypher query. If not, there is no need for a Cypher query.

Step 3: Review Schema for Relationship Types and Properties
Check the schema for allowed relationship types and properties. Only use relationships and properties specified in the schema for constructing the Cypher query.

Step 4: Construct the Cypher Query
Based on the analysis of the user's question and the schema, create the Cypher query. Ensure it reflects the user’s needs—such as finding mixtures, substances, properties, or combinations thereof. The query should filter based on the given properties (like melting point, proportion, etc.) or any other requirements in the question.

Step 5: Return the Output
If a Cypher query is needed, return the Cypher query in JSON format. If no query is needed, simply return "no" in the same format.
"""

ANSWER_SYSTEM_PROMPT = """
Task: Generate a response to the user's question based on the Cypher query result.
The database is a knowledge graph about Deep Eutectic Solvent(DES) mixtures and their substances.
Instructions:
Use the provided Cypher query result to answer the user's question.
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a response.
Do not include any text except the generated response.
"""

FEEWSHOT_EXAMPLES = """
Example 1:
Question: 寻找 meltingPoint 在 [300, 400] K 范围内的所有配方
Cypher Query:
MATCH (m:Mixture)
WHERE m.meltingPoint >= $minMP
  AND m.meltingPoint <= $maxMP
RETURN m

Example 2:
Question: 寻找包含 Glycerin，且 meltingPoint 在 [290, 350] K 范围内的配方
Cypher Query:
MATCH (s:Substance { pubChemName: $substanceName })<-[:hasSubstance]-(m:Mixture)
WHERE m.meltingPoint >= $minMP
  AND m.meltingPoint <= $maxMP
RETURN m

Example 3:
Question: 寻找包含 Sodium Chloride 和 Calcium Chloride，且 meltingPoint 在 [400, 600]，TPSA 在 [0, 100] 的配方
Cypher Query:
WITH ["Sodium Chloride", "Calcium Chloride"] AS targetSubstances
MATCH (m:Mixture)-[:hasSubstance]->(s:Substance)
WHERE s.pubChemName IN targetSubstances
WITH m, COUNT(DISTINCT s) AS matchedCount, targetSubstances
WHERE matchedCount = SIZE(targetSubstances)
  AND m.meltingPoint >= $minMP
  AND m.meltingPoint <= $maxMP
  AND m.TPSA >= $minTPSA
  AND m.TPSA <= $maxTPSA
RETURN m

Example 4:
Question: 寻找 meltingPoint 在 [300, 400] K 范围内最低 melting point 的配方
Cypher Query:
MATCH (m:Mixture)
WHERE m.meltingPoint >= $minMP
  AND m.meltingPoint <= $maxMP
WITH m
ORDER BY m.meltingPoint ASC
LIMIT 1
RETURN m

Example 5:
Question: 查找所有曾研究过 Glycerin 这个物质的文章
Cypher Query:
MATCH (s:Substance { pubChemName: $substanceName })<-[:hasSubstance]-(m:Mixture)-[:isReportedIn]->(a:Article)
RETURN DISTINCT a

Example 6:
Question: 数据库里 meltingPoint 最低的配方是哪一个？它包含哪些物质及对应配比？
Cypher Query:
MATCH (m:Mixture)-[rel:hasSubstance]->(s:Substance)
WHERE m.meltingPoint IS NOT NULL
WITH m, COLLECT({ name: s.pubChemName, proportion: rel.proportion }) AS composition
ORDER BY m.meltingPoint ASC
LIMIT 1
RETURN m.mixtureID AS mixture,
m.meltingPoint AS meltingPoint,
composition

Example 7:
Question: 在 [Sodium Chloride, Calcium Chloride] 这个二元体系里，哪个配比(配方)的 meltingPoint 最低？
Cypher Query:
WITH ["Sodium Chloride", "Calcium Chloride"] AS targetSystem
MATCH (m:Mixture)-[rel:hasSubstance]->(s:Substance)
WHERE s.pubChemName IN targetSystem
WITH m, COUNT(DISTINCT s) AS matchedCount
WHERE matchedCount = SIZE(targetSystem)
WITH m
ORDER BY m.meltingPoint ASC
LIMIT 1
MATCH (m)-[rel:hasSubstance]->(sub:Substance)
RETURN m.mixtureID AS mixtureID,
m.meltingPoint AS meltingPoint,
COLLECT({ name: sub.pubChemName, proportion: rel.proportion }) AS composition
"""

EXAMPLE_OUTPUT_PROMPT = """
Example Output for yes:
{{"thought_process": "explain the reasoning behind the query construction","use_cypher": "yes","cypher_query": "your cypher query here"}}
Example Output for no:
{{"thought_process": "explain why a Cypher query is not needed","use_cypher": "no"}}
""" 

CONVERT_CYPHER_SYSTEM_PROMPT = """
You are a helpful assistant that converts Cypher queries into graph-oriented queries, focusing on returning paths (nodes and relationships) for visualization. Use the provided schema and result to construct the query.

Your task is to transform the given Cypher query into one that will return paths, which can be visualized in a graph, rather than just returning tables of results. You should ensure that:
- Each query involves paths represented as nodes and relationships.
- If there are multiple relationships or conditions, you should break them into separate path variables if necessary.
- If there are `MATCH` clauses involving multiple relationships, you should separate them into distinct path variables to maintain the correct structure.

Instructions:
1. Identify the relationships in the query.
2. Break the query into multiple paths if it contains multiple relationships that need to be visualized.
3. Ensure the query focuses on showing the structure of the graph (paths), not just data points or aggregated results.
4. Use the `MATCH` clause to capture paths of nodes connected by relationships.
5. Return the query with the `RETURN` clause showing the path(s).
6. Please return the query in JSON format.

### Few-shot examples:

Example 1:
Input:
MATCH (m:Mixture)-[r:hasSubstance]->(s:Substance)
RETURN m, s
Output:
{{"cypher_query": "MATCH p=(m:Mixture)-[r:hasSubstance]->(s:Substance) RETURN p"}}

Example 2:
Input:
MATCH (m:Mixture)-[r1:hasSubstance]->(s:Substance {{pubChemName: 'Urea'}})
MATCH (m)-[r2:hasSubstance]->(other:Substance)
WHERE other.pubChemName <> 'Urea'
RETURN m, s, other
Output:
{{"cypher_query": "MATCH p1=(m:Mixture)-[r1:hasSubstance]->(s:Substance {{pubChemName: 'Urea'}}) MATCH p2=(m)-[r2:hasSubstance]->(other:Substance) WHERE other.pubChemName <> 'Urea' RETURN p1, p2"}}

Example 3:
Input:
MATCH (m:Mixture)-[r1:hasSubstance]->(s:Substance {{pubChemName: 'Urea'}})-[r2:hasSubstance]->(other:Substance)
RETURN m, s, other
Output:
{{"cypher_query": "MATCH p1=(m:Mixture)-[r1:hasSubstance]->(s:Substance {{pubChemName: 'Urea'}}) MATCH p2=(m)-[r2:hasSubstance]->(other:Substance) RETURN p1, p2"}}

Now, you will be given a Cypher query. Please convert it to focus on showing the graph paths.
"""

