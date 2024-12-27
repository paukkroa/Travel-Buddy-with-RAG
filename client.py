import os
import time
from langchain.prompts import ChatPromptTemplate

# Silence warnings
import warnings
warnings.filterwarnings('ignore')
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# Import functions from other files
from create_or_update_db import create_or_update_chroma, clear_database
from data_loader import load_documents, split_documents
from rag import query_and_response
from llm import prompt_model
from logger import get_logger
from llm_utils import sys_prompts, PROMPT_TEMPLATE
import db
import hashlib

class TravelBuddyCLI():
    def __init__(self, 
                 model_type = "gemini", 
                 model_name = "gemini-2.0-flash-exp",
                 chroma_path = "chroma",
                 data_path = "data",
                 chunk_size = 500,
                 chunk_overlap = 100):
        self.model_type = model_type
        self.client_name = "TravelBuddy"
        self.model_name = model_name
        self.chroma_path = chroma_path
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cli_commands = {"/bye": "Exit the CLI.",
                             "/help": "List all commands.",
                             "/destination": "Ask for a destination recommendation.",
                             "/travel_tips": "Ask for travel tips.",
                             "/query": "Ask a general question.",
                             "/update_db": "Update the data and database.",
                             }
        self.logger = get_logger(f"{self.client_name}")
        self.last_checked_path = "last_checked.txt"
        self.query_k_results = 10
        self.conn = db.create_connection()

    def _get_last_modified_time(self, directory):
        """
        Get the maximum last modified time of all files in a directory.
        """
        return max(os.path.getmtime(os.path.join(directory, f)) for f in os.listdir(directory))

    def _load_data(self):
        """
        Load data from the data path and split it into chunks.
        """
        self.logger.info("Loading data")
        documents = load_documents(self.data_path)
        self.logger.info("Splitting documents")
        self.chunks = split_documents(documents = documents, 
                                      chunk_size = self.chunk_size, 
                                      chunk_overlap = self.chunk_overlap)

    def _update_vector_db(self):
        """
        Update the chroma database with the new chunks.
        """
        self.logger.info("Updating database")
        create_or_update_chroma(chroma_path = self.chroma_path, 
                                chunks = self.chunks, 
                                model = self.model_type)

    def _clear_vector_db(self):
        """
        Empty the chroma database.
        """
        self.logger.info("Clearing database")
        clear_database()

    def _export_last_checked_time(self):
        """
        Export the last checked time.
        """
        with open(self.last_checked_path, "w") as f:
            f.write(str(self.last_checked_time))
            f.close()
        
    def _update_last_checked_time(self):
        """
        Get the last checked time.
        """
        try:
            with open(self.last_checked_path, "r") as f:
                self.last_checked_time = float(f.read())
                f.close()
        except FileNotFoundError:
            self.last_checked_time = None

    def prepare_chat(self):
        if self.conn is None:
            self.logger.error("Could not connect to the database, chat history is not available.")
            self.username = "You"
        
        # Create tables if they do not exist
        db.create_tables(self.conn)

        # Get the chat ID
        self.chat_id = db.get_most_recent_chat_id(self.conn)

        if self.chat_id is None:
            self.username = input(f"Welcome to {self.client_name}! Before we begin, please enter your name:\nYour name: ")
            self.chat_id = hashlib.md5(self.username.encode()).hexdigest()
            self.chat_description = f"Chat with {self.username}"
            self.context_type = "last_n"
            self.context_length = 10
            db.create_chat(self.conn, self.chat_id, self.username, self.chat_description, self.context_type, self.context_length)
        else:
            chat_info = db.get_chat_info(self.conn, self.chat_id)
            self.chat_id = chat_info["chat_id"]
            self.username = chat_info["chat_name"]
            self.chat_description = chat_info["chat_description"]
            self.context_type = chat_info["context_type"]
            self.context_length = chat_info["context_length"]
            print(f"TravelBuddy: Welcome back, {self.username}!")

    def prepare_data(self):
        self._update_last_checked_time()
        # Check if the data has been modified since the last check.
        last_modified_time = self._get_last_modified_time(directory = self.data_path)
        if self.last_checked_time is None or last_modified_time > self.last_checked_time:
            self._load_data()
            self._update_vector_db()
            self.last_checked_time = time.time()
            self._export_last_checked_time()
        else:
            self.logger.info("Data has not been modified since last start.")

    def main_loop(self):
        user_input = ""

        # Start main loop
        while user_input != "/bye":
            # Init variables and get input
            response = ""
            chat_history = db.get_chat_messages(self.conn, self.chat_id)
            user_input = input(f"{self.username}: ")

            # Chatting with the model without RAG
            if user_input not in self.cli_commands:
                # Prepare prompt with chat history
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(message=user_input, chat_history=chat_history)
                # Get response from the model
                response = prompt_model(prompt, self.model_type, self.model_name, sys_prompts["generic"])
                print(f"{self.client_name}: {response}")

            # Use RAG to query the database in search of a destination
            elif user_input == "/destination":
                query_text = input(f"{self.client_name}: What kind of destination are you looking for?\n{self.username}: ")
                response, formatted_response = query_and_response(query_text, 
                                                                  self.model_type, 
                                                                  self.model_name, 
                                                                  sys_prompt=sys_prompts["destination"], 
                                                                  k=self.query_k_results,
                                                                  chat_history=chat_history)
                print(f"{self.client_name}: {formatted_response}")

            # Use RAG to query the database for travel tips
            elif user_input == "/travel_tips":
                query_text = input(f"{self.client_name}: What travel tips do you need?\n{self.username}: ")
                response, formatted_response = query_and_response(query_text, 
                                                                  self.model_type, 
                                                                  self.model_name, 
                                                                  sys_prompt=sys_prompts["travel_tips"], 
                                                                  k=self.query_k_results,
                                                                  chat_history=chat_history)
                print(f"{self.client_name}: {formatted_response}")

            # Use RAG to query the database for a general question
            elif user_input == "/query":
                query_text = input(f"{self.client_name}: Enter your query.\n{self.username}: ")
                response, formatted_response = query_and_response(query_text, 
                                                                  self.model_type, 
                                                                  self.model_name, 
                                                                  sys_prompt=sys_prompts["generic_rag"],
                                                                  k=self.query_k_results, 
                                                                  chat_history=chat_history)
                print(f"{self.client_name}: {formatted_response}")
            
            # Update the data and database
            elif user_input == "/update_db":
                self._load_data()
                self._update_vector_db()
                self.last_checked_time = time.time()
                self._export_last_checked_time()
                response = "Database updated."
                print(f"{self.client_name}: {response}")

            # List all commands
            elif user_input == "/help":
                print(f"{self.client_name}: Here are all of the commands available:\n")
                for command, description in self.cli_commands.items():
                    print(f"{command}: {description}")

            # Exit the CLI
            elif user_input == "/bye":
                break

            # Add the user input and response to the chat history
            db.add_message_to_chat(self.conn, self.chat_id, self.username, user_input)
            db.add_message_to_chat(self.conn, self.chat_id, self.client_name, response)

    def start(self):
        # Prepare the CLI
        self.logger.info(f"Starting {self.client_name} CLI")
        self.prepare_data()
        self.prepare_chat()
        
        # Start the main loop
        self.main_loop()
        
        # End the CLI when main loop is exited
        self.end()
    
    def end(self):
        """
        Exit the CLI.
        """
        print(f"Exiting {self.client_name} CLI.")
        db.close_connection(self.conn)
        return

        

if __name__ == "__main__":
    client = TravelBuddyCLI()
    client.start()
    