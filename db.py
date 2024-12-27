import sqlite3

from logger import get_logger

logger = get_logger(__name__)

def create_connection(db_file: str = "chat.db") -> sqlite3.Connection:
    """
    Create a database connection to the SQLite database specified by db_file
    
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logger.info(f"Connected to {db_file}")
    except sqlite3.Error as e:
        logger.error(e)
    
    return conn

def close_connection(conn: sqlite3.Connection) -> None:
    conn.close()

def create_tables(conn: sqlite3.Connection) -> None:
    # Chats
    conn.execute('''
    CREATE TABLE IF NOT EXISTS D_CHAT (
        chat_id TEXT PRIMARY KEY,
        chat_name TEXT NOT NULL,
        chat_description TEXT,
        iby TEXT DEFAULT 'system',
        uby TEXT DEFAULT 'system',
        idate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        udate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    ''')

    # Chat settings
    conn.execute('''
    CREATE TABLE IF NOT EXISTS D_CHAT_SETTINGS (
        chat_id TEXT NOT NULL,
        context_type TEXT DEFAULT 'last_n',
        context_length INTEGER DEFAULT 10,
        iby TEXT DEFAULT 'system',
        uby TEXT DEFAULT 'system',
        idate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        udate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES D_CHAT(chat_id)
    );
    ''')

    # Chat history
    conn.execute('''
    CREATE TABLE IF NOT EXISTS F_CHAT_HISTORY (
        chat_id TEXT NOT NULL,
        sender TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        iby TEXT DEFAULT 'system',
        uby TEXT DEFAULT 'system',
        idate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        udate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (chat_id) REFERENCES D_CHAT(chat_id)
    );
    ''')
    conn.commit()

def drop_tables(conn: sqlite3.Connection) -> None:
    conn.execute('''
    DROP TABLE IF EXISTS D_CHAT;
    ''')
    conn.execute('''
    DROP TABLE IF EXISTS D_CHAT_SETTINGS;
    ''')
    conn.execute('''
    DROP TABLE IF EXISTS F_CHAT_HISTORY;
    ''')
    conn.commit()

def create_chat(conn: sqlite3.Connection, 
                chat_id: str, 
                chat_name: str, 
                chat_description: str = None,
                context_type: str = 'last_n',
                context_length: int = 10) -> None:
    """
    Create a chat.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :param chat_name: Chat name
    :param chat_description: Chat description
    """
    conn.execute('''
    INSERT INTO D_CHAT (chat_id, chat_name, chat_description) VALUES (?, ?, ?)
    ''', (chat_id, chat_name, chat_description))

    conn.execute('''
    INSERT INTO D_CHAT_SETTINGS (chat_id, context_type, context_length) VALUES (?, ?, ?)
    ''', (chat_id, context_type, context_length))

    conn.commit()

def get_chat(conn: sqlite3.Connection, chat_id: str) -> tuple:
    """
    Get a chat.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :return: Chat info as tuple
    """
    cursor = conn.execute('''
    SELECT chat_id, chat_name, chat_description FROM D_CHAT WHERE chat_id = ?
    ''', (chat_id,))
    
    chat = cursor.fetchone()
    
    return chat


def add_message_to_chat(conn: sqlite3.Connection, chat_id: str, sender: str, message: str) -> None:
    """
    Add a message to the chat history.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :param sender: Sender of the message
    :param message: Message content
    """
    conn.execute('''
    INSERT INTO F_CHAT_HISTORY (chat_id, sender, message) VALUES (?, ?, ?)
    ''', (chat_id, sender, message))

    conn.commit()

def get_all_chat_messages(conn: sqlite3.Connection, chat_id: str) -> list:
    """
    Get all messages from a chat.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :return: List of messages
    """
    cursor = conn.execute('''
    SELECT sender, message, timestamp FROM F_CHAT_HISTORY WHERE chat_id = ?
    ''', (chat_id,))
    
    messages = cursor.fetchall()
    
    return messages

def get_last_n_messages(conn: sqlite3.Connection, chat_id: str, n: int) -> list:
    """
    Get the last n messages from a chat.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :param n: Number of messages
    :return: List of messages
    """
    cursor = conn.execute('''
    SELECT sender, message, timestamp FROM F_CHAT_HISTORY WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?
    ''', (chat_id, n))
    
    messages = cursor.fetchall()
    
    return messages

def get_chat_settings(conn: sqlite3.Connection, chat_id: str) -> dict:
    """
    Get the chat settings.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :return: Dictionary of chat settings
    """
    cursor = conn.execute('''
    SELECT context_type, context_length FROM D_CHAT_SETTINGS WHERE chat_id = ?
    ''', (chat_id,))
    
    settings = cursor.fetchone()
    
    return settings

def get_chat_messages(conn: sqlite3.Connection, chat_id: str) -> str:
    """
    Get chat messages based on the current setting
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :return: String of the messages
    """
    settings = get_chat_settings(conn, chat_id)

    context_type = settings[0]
    context_length = settings[1]

    if context_type == "last_n":
        messages = get_last_n_messages(conn, chat_id, context_length)
    elif context_type == "all":
        messages = get_all_chat_messages(conn, chat_id)
    else: 
        messages = []    

    messages_str = "\n".join([f"At {msg[2]}, {msg[0]} said: {msg[1]}" for msg in messages])
    return messages_str

def get_most_recent_chat_id(conn: sqlite3.Connection) -> str:
    """
    Get the chat ID with the most recent message.
    
    :param conn: Connection object
    :return: Chat ID
    """
    cursor = conn.execute('''
    SELECT chat_id FROM F_CHAT_HISTORY ORDER BY timestamp DESC LIMIT 1
    ''')
    
    chat_id = cursor.fetchone()
    
    return str(chat_id[0]) if chat_id else None

def get_chat_info(conn: sqlite3.Connection, chat_id: str) -> dict:
    """
    Get chat info.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    :return: Dictionary of chat info
    """
    chat = get_chat(conn, chat_id)
    settings = get_chat_settings(conn, chat_id)

    chat_info = {
        "chat_id": chat[0],
        "chat_name": chat[1],
        "chat_description": chat[2],
        "context_type": settings[0],
        "context_length": settings[1]
    }

    return chat_info

def get_all_chatnames(conn: sqlite3.Connection) -> list:
    """
    Get all chat names.
    
    :param conn: Connection object
    :return: List of chat names
    """
    cursor = conn.execute('''
    SELECT chat_name FROM D_CHAT
    ''')
    
    chat_names = cursor.fetchall()
    
    return [chat_name[0] for chat_name in chat_names]

def delete_user_and_chat(conn: sqlite3.Connection, chat_id: str) -> None:
    """
    Delete a user and their chat.
    
    :param conn: Connection object
    :param chat_id: Chat ID
    """
    conn.execute('''
    DELETE FROM F_CHAT_HISTORY WHERE chat_id = ?
    ''', (chat_id,))
    
    conn.execute('''
    DELETE FROM D_CHAT_SETTINGS WHERE chat_id = ?
    ''', (chat_id,))
    
    conn.execute('''
    DELETE FROM D_CHAT WHERE chat_id = ?
    ''', (chat_id,))
    
    conn.commit()