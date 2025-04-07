# DesAgent.py
import time
# start_time = time.time()
import os
# import torch
from operator import itemgetter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

import re
from typing import List, Dict, Any

import pandas as pd
import json
from json_repair import repair_json
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
# from agent_prompt import *
from prompts import (
    CYHPER_SYSTEM_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    FEEWSHOT_EXAMPLES,
    EXAMPLE_OUTPUT_PROMPT,
    CONVERT_CYPHER_SYSTEM_PROMPT
)

# Environment variables setup
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "KG Query Agent"

NEO4J_URI = "neo4j+s://70d99939.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "H3wGAoxB6YBnD3paWrcAnDDZAKFBT8hR3kDtAK7nZmE"

BASE_URL = None
# BASE_URL = "https://service-56tr1g58-1317694151.usw.apigw.tencentcs.com/v1"

CYPHER_CLAUSE_KEYWORDS = [
    'MATCH',
    'OPTIONAL MATCH',
    'WHERE',
    'RETURN',
    'WITH',
    'CREATE',
    'MERGE',
    'DELETE',
    'SET',
    'UNWIND',
    'ORDER BY',
    'SKIP',
    'LIMIT',
    'FOREACH',
    'CALL',
    'DETACH DELETE',
    'REMOVE'
]

CLAUSE_PATTERN = '|'.join([re.escape(keyword) for keyword in CYPHER_CLAUSE_KEYWORDS])

# end_time = time.time()
# print(f"Time taken for imports: {end_time - start_time:.2f}s")


class DesAgent:
    """Description Agent for querying Neo4j graph using LangChain."""
    
    def __init__(self, llm_model_name="gpt-4o", session_id=None):
        """
        Initialize the DesAgent with LLM model and session details.

        Args:
            llm_model_name (str): Name of the language model to use.
            session_id (str, optional): Session identifier. Defaults to "global".
        """
        self.llm_model_name = llm_model_name
        self.session_id = session_id or "global"  # fallback
        self.log_dir = "chat_logs"
        self.log_file = f"./{self.log_dir}/chat_log_{self.session_id}.txt"
        self.CHAT_HISTORY = ChatMessageHistory()
        self.CHAT_HISTORY_FILE_PATH = "chat_history/chat_history.txt"
        try:
            self.graph = Neo4jGraph(
                url=NEO4J_URI,
                username=NEO4J_USER,
                password=NEO4J_PASSWORD,
                # driver_config={"notifications_min_severity":"WARNING","notifications_disabled_classifications": ["UNRECOGNIZED","DEPRECATED"]},
                enhanced_schema=True
            )
            self.schema = self.graph.schema
        except Exception as e:
            print(f"[Error initializing Neo4j connection]: {e}")
            self.graph = None
            self.schema = "No schema available due to connection error."

        self.fewshot_examples = FEEWSHOT_EXAMPLES
        self.example_output_prompt = EXAMPLE_OUTPUT_PROMPT
        self.cypher_system_prompt = CYHPER_SYSTEM_PROMPT
        self.answer_system_prompt = ANSWER_SYSTEM_PROMPT
        self.convert_cypher_system_prompt = CONVERT_CYPHER_SYSTEM_PROMPT
        self.schema = f"Schema: {self.schema}"
        
        self.cypher_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.cypher_system_prompt),
                ("system", "{fewshot_examples}"),
                ("system", self.example_output_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", self.schema),    
                ("human", "{question}"),
            ]
        )
        self.cypher_llm = ChatOpenAI(model=self.llm_model_name,temperature=0,base_url=BASE_URL,model_kwargs={"response_format": {"type": "json_object"}})
        self.answer_llm = ChatOpenAI(model=self.llm_model_name,temperature=0,base_url=BASE_URL)
        self.cypher_chain = (
            {"question": itemgetter("question"), "chat_history": itemgetter("chat_history"), "fewshot_examples": itemgetter("fewshot_examples")}
            | self.cypher_agent_prompt
            | self.cypher_llm 
            | JsonOutputParser())
        
        self.fix_cypher_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Fix the cypher query based on the error message. Only return the fixed cypher query."),
                ("human", "Question: {question}"),
                ("human", "Schema: {schema}"),
                ("human", "Error message: {error_message}"),
                ("human", "Wrong cypher query: {cypher_query}"),
            ]
        )
        self.fix_cypher_query_chain = (
            {"question": itemgetter("question"), "schema": itemgetter("schema"), "error_message": itemgetter("error_message"), "cypher_query": itemgetter("cypher_query")}
            | self.fix_cypher_query_prompt
            | self.cypher_llm
            | StrOutputParser()
        )

        self.convert_cypher_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.convert_cypher_system_prompt),
                ("system", "{schema}"),
                ("human", "Cypher Query: {cypher_query}, \nResult: {result}"),
            ]
        )
        self.convert_cypher_chain = (
            {"cypher_query": itemgetter("cypher_query"), "result": itemgetter("result"),"schema": itemgetter("schema")}
            | self.convert_cypher_prompt
            | self.cypher_llm
            | JsonOutputParser()
        )
        self.cypher_clause_keywords = CYPHER_CLAUSE_KEYWORDS
        self.clause_pattern = CLAUSE_PATTERN

        self.start_session_log()
        
        # Initialize a list to store processed results for each query
        self.processed_results = []

    def start_session_log(self):
        """
        Start logging the session by recording the start time.
        """
        self.session_log = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "end_time": None,
            "messages": []
        }
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"Session start time: {self.session_log['start_time']}\n")
        except Exception as e:
            print(f"[Error writing to log file]: {e}")

    def log_message(self, role, content):
        """
        Log a message from the user or AI.

        Args:
            role (str): The role of the message sender ('user' or 'ai').
            content (str): The content of the message.
        """
        self.session_log["messages"].append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "role": role,
            "content": content
        })
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"{role.capitalize()}: {content}\n")
        except Exception as e:
            print(f"[Error writing message to log file]: {e}")

    def save_session_log(self, log_filepath=None):
        """
        Save the session log by recording the end time.

        Args:
            log_filepath (str, optional): Path to save the log file. Defaults to self.log_file.
        """
        if not self.session_log["end_time"]:
            self.session_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            final_log_file = log_filepath if log_filepath else self.log_file
            try:
                with open(final_log_file, "a", encoding='utf-8') as f:
                    f.write(f"Session end time: {self.session_log['end_time']}\n\n")
            except Exception as e:
                print(f"[Error writing end time to log file]: {e}")

    def fix_cypher_query(self, cypher_query, error_message, question):
        """
        Attempt to fix a faulty Cypher query based on the error message.

        Args:
            cypher_query (str): The original Cypher query.
            error_message (str): The error message returned from Neo4j.
            question (str): The user's original question.

        Returns:
            str or None: The fixed Cypher query or None if unable to fix.
        """
        try:
            return self.fix_cypher_query_chain.invoke({
                "cypher_query": cypher_query, 
                "error_message": error_message, 
                "question": question, 
                "schema": self.schema
            })
        except Exception as e:
            self.log_message("ai", f"Error while fixing cypher query: {e}")
            return None
    
    def convert_query_result(self, result, result_type="json"):
        """
        Convert the query result to a pandas DataFrame.
        """
        if result_type == "json":
            return repair_json(result)
        elif result_type == "md":
            return pd.DataFrame(result).to_markdown(index=False)

    def query_graph_with_retry(self, cypher_query, retry_count=3, question=None):
        """
        Query the Neo4j graph with a retry mechanism.

        Args:
            cypher_query (str): The Cypher query to execute.
            retry_count (int, optional): Number of retry attempts. Defaults to 3.
            question (str, optional): The user's original question.

        Returns:
            Any or None: The query result or None if unsuccessful.
        """
        if not self.graph:
            self.log_message("ai", "No graph connection available.")
            return None
        for i in range(retry_count):
            try:
                self.log_message("ai", f"Cypher query: {cypher_query}")
                self.CHAT_HISTORY.add_ai_message(f"Cypher query: {cypher_query}")
                result = self.graph.query(cypher_query)
                result = self.convert_query_result(result, result_type="md")
                return result
            except Exception as e:
                print(f"Error: {e}, please fix the cypher query and try again.")
                self.log_message("ai", f"Error: {e}, please fix the cypher query and try again.")
                self.CHAT_HISTORY.add_ai_message(f"Error: {e}, please fix the cypher query and try again.")
                cypher_query = self.fix_cypher_query(cypher_query, error_message=e, question=question)
                if i == retry_count - 1:
                    return None
        return None
    
    def cypher_query_to_path(self, cypher_query,result,question):
        """
        Convert the cypher query to focus on showing the graph paths.

        Args:
            cypher_query (str): The original Cypher query.
            result (Any): The result of the cypher query.
            question (str): The user's original question.

        Returns:
            str: The converted Cypher query.
        """
        converted_cypher_query = self.convert_cypher_chain.invoke({"cypher_query": cypher_query, "result": result, "schema": self.schema})
        converted_cypher_query = converted_cypher_query["cypher_query"]
        result = self.query_graph_with_retry(converted_cypher_query, retry_count=3, question=question)

        if result is None:
            msg = "Error: No results found. Please try with another cypher query."
            self.log_message("ai", msg)
            self.CHAT_HISTORY.add_ai_message(msg)
            return msg
        else:
            return result,converted_cypher_query
    
    def parse_query_paths(self,query: str) -> Dict[str, List[str]]:
        """
        Parses the Cypher query to extract the structure of each path in the RETURN clause.
        Returns a dictionary mapping each path variable to a list representing the sequence
        of node labels and relationship types.
        """
        paths = {}
        variable_label_map = {}  # Tracks variable to label mappings

        # Normalize whitespace for consistent regex matching
        normalized_query = ' '.join(query.strip().split())

        # Extract RETURN clause
        return_clause_match = re.search(r'\bRETURN\b\s+(.+)', normalized_query, re.IGNORECASE)
        if not return_clause_match:
            raise ValueError("No RETURN clause found in the query.")

        return_vars = return_clause_match.group(1).split(',')
        return_vars = [var.strip() for var in return_vars]

        # Extract all MATCH clauses
        # This regex captures 'MATCH' or 'OPTIONAL MATCH' followed by anything until the next clause keyword or end of string
        match_clauses = re.findall(
            rf'(?:MATCH|OPTIONAL MATCH)\s+(.*?)(?=\b(?:{CLAUSE_PATTERN})\b|$)',
            normalized_query,
            re.IGNORECASE
        )

        if not match_clauses:
            raise ValueError("No MATCH clauses found in the query.")

        for match_clause in match_clauses:
            # Remove any trailing WHERE clauses or other filters within the MATCH clause
            match_clause_clean = re.split(r'\bWHERE\b', match_clause, flags=re.IGNORECASE)[0].strip()

            # Extract the path variable name and the path pattern
            # Pattern: path_var=pattern or just pattern without assignment
            path_var_match = re.match(r'(\w+)\s*=\s*(.+)', match_clause_clean)
            if not path_var_match:
                # Handle MATCH without assignment, e.g., MATCH (n:Label)-[:REL]->(m:Label)
                # Assign a default variable name
                path_var = f"path_{len(paths)+1}"
                path_pattern = match_clause_clean
            else:
                path_var = path_var_match.group(1)
                path_pattern = path_var_match.group(2)

            # Extract nodes and relationships using regex
            # Nodes are within round brackets ()
            # Relationships are within square brackets []
            node_pattern = r'\(([^()]+)\)'
            relationship_pattern = r'\[([^()\[\]]+)\]'

            # Find all nodes and relationships in order
            nodes = re.findall(node_pattern, path_pattern)
            relationships = re.findall(relationship_pattern, path_pattern)

            if not nodes:
                print(f"Warning: No nodes found in MATCH clause: {match_clause_clean}")
                continue
            if not relationships:
                print(f"Warning: No relationships found in MATCH clause: {match_clause_clean}")
                # It's possible to have MATCH clauses without relationships
                # Handle accordingly if needed

            # Extract node labels by splitting on ':' and taking the second part
            node_labels = []
            for node in nodes:
                parts = node.split(':')
                if len(parts) >= 2:
                    variable = parts[0].strip()
                    label_and_props = parts[1].strip()
                    # Extract label before any space or property
                    label = re.split(r'\s|\{', label_and_props)[0]
                    node_labels.append(label)
                    # Update variable_label_map if variable is present
                    if variable:
                        variable_label_map[variable] = label
                elif len(parts) == 1:
                    # Node without label, possibly just a variable
                    variable = parts[0].strip()
                    label = variable_label_map.get(variable, 'Unknown')  # Retrieve label if exists
                    node_labels.append(label)
                    if variable:
                        variable_label_map[variable] = label
                else:
                    node_labels.append('Unknown')

            # Extract relationship types by splitting on ':' and taking the second part
            rel_types = []
            for rel in relationships:
                parts = rel.split(':')
                if len(parts) >= 2:
                    rel_type = parts[1].strip().split(']')[0]  # Removes any trailing characters if present
                    rel_types.append(rel_type)
                else:
                    rel_types.append('UNKNOWN_RELATIONSHIP')  # Default type if not specified

            # Reconstruct the sequence: node, relationship, node, relationship, ...
            sequence = []
            for i in range(len(rel_types)):
                # Append node label
                if i < len(node_labels):
                    sequence.append(node_labels[i])
                else:
                    sequence.append('Unknown')
                # Append relationship type
                sequence.append(rel_types[i])
            # Append the last node label if exists
            if len(node_labels) > len(rel_types):
                sequence.append(node_labels[-1])

            # Assign the sequence to the path variable
            paths[path_var] = sequence

        # # Debug: Print parsed paths
        # print("\n[DEBUG] Parsed Paths:")
        # for var, seq in paths.items():
        #     print(f"  {var}: {seq}")

        return paths
    # Function to process the Neo4j query results based on the parsed paths
    def process_results(self, paths: Dict[str, List[str]], results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes the Neo4j query results dynamically based on the provided path structures.
        Returns a list of processed results with labeled nodes and relationships.
        """
        processed = []
        
        for record in results:
            processed_record = {}
            for path_var, path_structure in paths.items():
                if path_var not in record:
                    continue
                path_data = record[path_var]
                sequence = path_structure
                nodes = []
                relationships = []
                
                # Iterate through the path data
                for index, element in enumerate(path_data):
                    if index % 2 == 0:
                        # Node
                        node_label = sequence[index]
                        node = {'label': node_label, 'properties': element}
                        nodes.append(node)
                    else:
                        # Relationship
                        rel_type = sequence[index]
                        relationship = {'type': rel_type}
                        relationships.append(relationship)
                
                # Combine nodes and relationships
                path_representation = []
                for i in range(len(nodes)):
                    path_representation.append({'n':nodes[i]})
                    if i < len(relationships):
                        path_representation.append({'r':relationships[i]})
                
                processed_record[path_var] = path_representation
            
            processed.append(processed_record)
        
        return processed


    def create_final_result_prompt_template(self, use_cypher, results):
        """
        Create the final prompt template based on query results.

        Args:
            use_cypher (str): Indicates whether a Cypher query was used.
            results (Any): The results from the Cypher query.

        Returns:
            str: The final prompt template.
        """
        template_parts = []

        if use_cypher == "yes" and results:
            template_parts.append("Cypher query result: {results}")
        elif use_cypher == "no":
            template_parts.append("No cypher query result needed, answer the question directly.")
        elif use_cypher == "yes" and results is None:
            template_parts.append("No results found, please try with another cypher query.")
        else:
            template_parts.append("There is an error in the cypher query.")
        
        return "\n\n".join(template_parts)

    def task_execution(self, question):
        """
        Execute the task based on the user's question.

        Args:
            question (str): The user's question.

        Yields:
            str: Responses or intermediate steps.
        """
        try:
            # Log the user's message
            self.log_message("user", question)
            self.CHAT_HISTORY.add_user_message(question)

            # Run the chain to decide if a cypher query is needed
            cypher_response = self.cypher_chain.invoke({
                "question": question,
                "chat_history": self.CHAT_HISTORY.messages,
                "fewshot_examples": self.fewshot_examples
            })
            use_cypher = cypher_response["use_cypher"]

            # Handle Cypher query execution
            if use_cypher == "yes":
                if "cypher_query" in cypher_response:
                    thought_process = cypher_response["thought_process"]
                    yield f"[Thought Process]\n{thought_process}\n\n"
                    self.CHAT_HISTORY.add_ai_message(f"[Thought Process]\n{thought_process}\n\n")
                    cypher_query = cypher_response["cypher_query"]
                    yield f"[Executing Cypher Query]\n{cypher_query}\n\n"
                    result = self.query_graph_with_retry(cypher_query, retry_count=3, question=question)
                    if result is None:
                        msg = "Error: No results found. Please try another query."
                        self.log_message("ai", msg)
                        self.CHAT_HISTORY.add_ai_message(msg)
                        yield msg + "\n\n"
                    else:
                        msg = f"Results found:\n{result}"
                        self.log_message("ai", msg)
                        self.CHAT_HISTORY.add_ai_message(msg)
                        yield msg + "\n\n"
                else:
                    msg = f"Error: No cypher query found in the response.\n{cypher_response}"
                    self.log_message("ai", msg)
                    self.CHAT_HISTORY.add_ai_message(msg)
                    yield msg + "\n\n"
                    result = None
            else:
                thought_process = cypher_response["thought_process"]
                yield f"[Thought Process]\n{thought_process}\n\n"
                self.CHAT_HISTORY.add_ai_message(f"[Thought Process]\n{thought_process}\n\n")
                result = None
                yield "No cypher query result needed, just answer directly.\n\n"
                self.CHAT_HISTORY.add_ai_message("No cypher query result needed, just answer directly.")

            # Prepare final response
            self.log_message("ai", f"Cypher query result: {result}")
            self.CHAT_HISTORY.add_ai_message(f"Cypher query result: {result}")
            final_result_prompt_template = self.create_final_result_prompt_template(use_cypher, result)
            answer_agent_prompt = ChatPromptTemplate.from_messages([
                ("system", self.answer_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("system", self.schema),
                ("human", final_result_prompt_template),
                ("human", "{question}"),
            ])

            current_chain = {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "results": itemgetter("results")
            }
            answer_agent = (
                current_chain
                | answer_agent_prompt
                | self.answer_llm
                | StrOutputParser()
            )
            chain_with_message_history = RunnableWithMessageHistory(
                answer_agent,
                lambda session_id: self.CHAT_HISTORY,
                input_messages_key='question',
                history_messages_key="chat_history",
            )

            respond_string = ""
            yield f"[Answer]\n"
            for chunk in chain_with_message_history.stream(
                {"question": question, "results": result},
                config={"configurable": {"session_id": self.session_id}}
            ):
                respond_string += chunk
                yield chunk

            # Log AI response
            self.log_message("ai", respond_string)
            self.CHAT_HISTORY.add_ai_message(respond_string)

            # Visualize the graph
            if use_cypher == "yes" and result:
                path_result,path_query = self.cypher_query_to_path(cypher_query, question,result)
                # print(f"cypher_query: {path_query}")
                yield f"\n[Visualize Query]\n{path_query}\n\n"
                path = self.parse_query_paths(path_query)
                processed_result = self.process_results(path, path_result)

                # Store the processed_result for visualization
                self.processed_results.append(processed_result)
            else:
                # Append None to maintain the order
                self.processed_results.append(None)

        except Exception as e:
            error_message = f"An error occurred while processing: {e}"
            self.log_message("ai", error_message)
            self.CHAT_HISTORY.add_ai_message(error_message)
            yield error_message

    def run(self):
        """
        Run the DesAgent, handling user input and responses.
        """
        print("System is ready. You can start asking questions.")
        while True:
            try:
                print("\n")
                question = input("Please enter your question (type 'q', 'quit', or 'exit' to stop): ")
                if question.lower() in ["q", "quit", "exit"]:
                    print("Exiting...")
                    self.save_session_log()
                    break
                for response in self.task_execution(question):
                    print(response, end="", flush=True)
                print("\n")
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    def get_latest_processed_result(self):
        """
        Retrieve the latest processed result.

        Returns:
            Dict[str, List[Dict[str, Any]]] or None: The latest processed result or None.
        """
        if self.processed_results:
            return self.processed_results[-1]
        return None

if __name__ == "__main__":
    agent = DesAgent()
    agent.run()