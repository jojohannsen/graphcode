import os
import pickle
import hashlib
import functools
import sqlite3
from typing import Callable, Any

def cache_llm_response_file(cache_dir: str = None):
    """
    Decorator for caching LLM responses using file system.
    
    Args:
        cache_dir: Directory for storing cached responses. If None, will use 'llm_cache' 
                  in the current working directory
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, prompt: str, **kwargs):
            # Get provider and model from the instance
            provider = getattr(self, 'provider', 'unknown')
            model = getattr(self, 'model', 'unknown')
            
            # Get the directory where this file (llm_cache.py) is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Create cache in the current directory
            actual_cache_dir = os.path.join(current_dir, "llm_cache")
            os.makedirs(actual_cache_dir, exist_ok=True)
            
            # Create a unique hash based on provider, model, and prompt
            cache_key = f"{provider}_{model}_{prompt}"
            prompt_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = os.path.join(actual_cache_dir, f"{prompt_hash}.pickle")
            
            # Check if we have a cached response
            if os.path.exists(cache_file):
                print(f"Cache hit for {provider}/{model} in {os.path.basename(os.path.dirname(actual_cache_dir))}: {prompt_hash}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # If no cache exists, call the original function and cache the result
            response = func(self, prompt, **kwargs)
            print(f"Cache miss for {provider}/{model} in {os.path.basename(os.path.dirname(actual_cache_dir))}: {prompt_hash}")
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
            
            return response
        return wrapper
    return decorator

def cache_llm_response_sqlite(db_path: str = None):
    """
    Decorator for caching LLM responses using SQLite database.
    
    Args:
        db_path: Path to SQLite database file. If None, will use 'llm_cache.db' 
                in the current working directory
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, prompt: str, **kwargs):
            # Get provider and model from the instance
            provider = getattr(self, 'provider', 'unknown')
            model = getattr(self, 'model', 'unknown')
            
            # Get the directory where this file (llm_cache.py) is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            actual_db_path = db_path or os.path.join(current_dir, "llm_cache.db")
            
            # Create a unique hash based on provider, model, and prompt
            cache_key = f"{provider}_{model}_{prompt}"
            prompt_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Initialize database and table if they don't exist
            with sqlite3.connect(actual_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS llm_cache (
                        cache_key TEXT PRIMARY KEY,
                        provider TEXT,
                        model TEXT,
                        prompt_hash TEXT,
                        response BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Check if we have a cached response
                cursor = conn.execute(
                    'SELECT response FROM llm_cache WHERE cache_key = ?',
                    (cache_key,)
                )
                row = cursor.fetchone()
                
                if row:
                    print(f"Cache hit for {provider}/{model} in DB: {prompt_hash}")
                    return pickle.loads(row[0])
            
            # If no cache exists, call the original function and cache the result
            response = func(self, prompt, **kwargs)
            print(f"Cache miss for {provider}/{model} in DB: {prompt_hash}")
            
            # Store in database
            with sqlite3.connect(actual_db_path) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO llm_cache (cache_key, provider, model, prompt_hash, response) VALUES (?, ?, ?, ?, ?)',
                    (cache_key, provider, model, prompt_hash, pickle.dumps(response))
                )
            
            return response
        return wrapper
    return decorator

class UnifiedLLM:
    """
    Unified LLM class that wraps ChatOpenAI, ChatAnthropic, etc.
    """
    def __init__(self, provider: str, model: str, cache_type: str = "file"):
        self.provider = provider
        self.model = model
        self.cache_type = cache_type
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(model=model)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(model=model)
        elif provider == "openrouter":
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1", model=model)
        else:
            raise ValueError(f"Unexpected provider, expected one of 'openai', 'anthropic', 'openrouter', got '{provider}'")
    
    def invoke(self, prompt: str, **kwargs):
        """Invoke the LLM with caching"""
        # Choose decorator based on cache type
        if self.cache_type == "sqlite":
            decorator = cache_llm_response_sqlite()
        else:  # default to file
            decorator = cache_llm_response_file()
        
        # Apply decorator to a method that calls the LLM
        @decorator
        def _invoke_with_cache(self, prompt: str, **kwargs):
            return self._llm.invoke(prompt, **kwargs)
        
        return _invoke_with_cache(self, prompt, **kwargs)
    

def make_llm(provider: str, model: str, cache_type: str = "file"):
    """
    Create a UnifiedLLM instance with specified caching method.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'openrouter')
        model: Model name
        cache_type: Cache type ('file' or 'sqlite'). Default is 'file'
    """
    return UnifiedLLM(provider, model, cache_type)