import os
import time

from create_or_update_db import create_or_update_chroma, clear_database
from data_loader import load_documents, split_documents
from embedding_function import get_embedding_function
from rag import query_and_response, query_only
from llm import prompt_model
from logger import get_logger
from llm_utils import sys_prompts

class TravelBuddyCLI():
    def __init__(self, 
                 model_type = "gemini", 
                 model_name = "gemini-2.0-flash-exp",
                 chroma_path = "chroma",
                 data_path = "data",
                 chunk_size = 500,
                 chunk_overlap = 100):
        self.model_type = model_type
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
        self.session_chat_history = []
        self.logger = get_logger("TravelBuddyCLI")
        self.last_checked_path = "last_checked.txt"
        self.query_k_results = 10

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

    def _update_db(self):
        """
        Update the chroma database with the new chunks.
        """
        self.logger.info("Updating database")
        create_or_update_chroma(chroma_path = self.chroma_path, 
                                chunks = self.chunks, 
                                model = self.model_type)

    def _clear_db(self):
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

    def start(self):
        self.logger.info("Starting TravelBuddy CLI")
        self._update_last_checked_time()
        # Check if the data has been modified since the last check.
        last_modified_time = self._get_last_modified_time(directory = self.data_path)
        if self.last_checked_time is None or last_modified_time > self.last_checked_time:
            self._load_data()
            self._update_db()
            self.last_checked_time = time.time()
            self._export_last_checked_time()
        else:
            self.logger.info("Data has not been modified since last start.")
        
        print("TravelBuddy CLI started. Enter '/bye' to exit.\nList more commands with '/help'.")
        user_input = ""
        while user_input != "/bye":
            user_input = input("You: ")
            if user_input not in self.cli_commands:
                response = prompt_model(user_input, self.model_type, self.model_name)
                self.session_chat_history.append(user_input)
                self.session_chat_history.append(response)
                print("TravelBuddy: " + response)
            elif user_input == "/destination":
                query_text = input("TravelBuddy: What kind of destination are you looking for?\nYou: ")
                query_and_response(query_text, self.model_type, self.model_name, sys_prompt=sys_prompts["destination"], k=self.query_k_results)
            elif user_input == "/travel_tips":
                query_text = input("TravelBuddy: What travel tips do you need?\nYou: ")
                query_and_response(query_text, self.model_type, self.model_name, sys_prompt=sys_prompts["travel_tips"], k=self.query_k_results)
            elif user_input == "/query":
                query_text = input("TravelBuddy: Enter your query.\nYou: ")
                query_and_response(query_text, self.model_type, self.model_name)
            elif user_input == "/update_db":
                self._load_data()
                self._update_db()
                self.last_checked_time = time.time()
                self._export_last_checked_time()
            elif user_input == "/help":
                print("TravelBuddy: Here are all of the commands available:\n")
                for command, description in self.cli_commands.items():
                    print(f"{command}: {description}")
            elif user_input == "/bye":
                break
        
        self.end()
    
    def end(self):
        """
        Exit the CLI.
        """
        print("Exiting TravelBuddy CLI.")
        return

        

if __name__ == "__main__":
    client = TravelBuddyCLI()
    client.start()
    