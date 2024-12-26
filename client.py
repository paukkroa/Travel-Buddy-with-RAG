CHROMA_PATH = "chroma"
DATA_PATH = "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

class TravelBuddyCLI():
    def __init__(self, 
                 model_type = "gemini", 
                 model_name = "gemini-2.0-flash-exp"):
        self.model_type = model_type
        self.model_name = model_name
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    